"""Concept index (plan §4.2): entities, names, frames, lexicon, events.

Each fact is stored ONCE as a frame event with roles, indexed by every
participant. Phrasings (relation words) map to roles of a frame through the
lexicon, so a fact seen in two phrasings is one event with two sources.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..world.api import FALSE, TRUE

TRUTH = {TRUE: 1, FALSE: 0}


@dataclass
class Entity:
    id: str
    context: str
    kind: str = "obj"
    provisional: bool = True


@dataclass
class LexEntry:
    """word -> (frame, sigma). sigma[i] = canonical position shown at surface position i
    (the last surface position is the word's output)."""
    word: str
    frame: str
    sigma: tuple[int, ...]
    support: int = 0
    contradictions: int = 0
    provisional: bool = True


@dataclass
class Event:
    id: str
    frame: str
    tup: tuple[int, ...]          # canonical tuple of entity indices (args..., out)
    sources: list[int] = field(default_factory=list)   # observation ids
    noise: bool = False


class Context:
    """Everything the learner knows inside one system context."""

    def __init__(self, cid: str):
        self.cid = cid
        self.ents: list[Entity] = []
        self.index: dict[str, int] = {}         # token -> entity index (within this context)
        self.tokens: list[str] = []             # entity index -> token (decoder)
        self.lex: dict[str, LexEntry] = {}
        self.frames: dict[str, dict] = {}        # frame id -> {"arity": k, "bool": bool}
        self.events: dict[str, dict[tuple, Event]] = defaultdict(dict)   # frame -> tup -> Event
        self.by_entity: dict[int, set[tuple[str, tuple]]] = defaultdict(set)
        self.obs: list[tuple[str, tuple[int, ...], int]] = []           # raw (word, args, out)
        # checker answers about surface claims: (word, args, out, result). Stored in surface
        # form so they stay true whatever the lexicon later decides.
        self.checked: list[tuple[str, tuple[int, ...], int, bool]] = []
        self.checked_why: list[str] = []         # why each check was asked (same index)

    def record_check(self, word: str, args: tuple, out: int, ok: bool, why: str) -> None:
        self.checked.append((word, tuple(args), out, bool(ok)))
        self.checked_why.append(why)

    # ---- entities and names -------------------------------------------------
    def ent(self, token: str) -> int:
        i = self.index.get(token)
        if i is None:
            i = len(self.ents)
            self.index[token] = i
            self.tokens.append(token)
            self.ents.append(Entity(f"{self.cid}:{i}", self.cid))
        return i

    @property
    def m(self) -> int:
        return len(self.ents)

    def value(self, token: str, boolean: bool) -> int:
        if boolean:
            return TRUTH[token]
        return self.ent(token)

    # ---- frames and lexicon ---------------------------------------------------
    def word_entry(self, word: str, arity: int, boolean: bool) -> LexEntry:
        e = self.lex.get(word)
        if e is None:
            fid = f"{self.cid}/F:{word}"
            self.frames[fid] = {"arity": arity, "bool": boolean}
            e = LexEntry(word, fid, tuple(range(arity + 1)))
            self.lex[word] = e
        return e

    def canon(self, entry: LexEntry, surface: tuple[int, ...]) -> tuple[int, ...]:
        k = len(surface)
        c = [0] * k
        for i, cp in enumerate(entry.sigma):
            c[cp] = surface[i]
        return tuple(c)

    def surface(self, entry: LexEntry, canon: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(canon[entry.sigma[i]] for i in range(len(canon)))

    def add_obs(self, word: str, args: tuple[int, ...], out: int, boolean: bool) -> Event:
        entry = self.word_entry(word, len(args), boolean)
        oid = len(self.obs)
        self.obs.append((word, args, out))
        tup = self.canon(entry, args + (out,))
        return self._put(entry.frame, tup, oid)

    def _put(self, frame: str, tup: tuple, oid: int) -> Event:
        ev = self.events[frame].get(tup)
        if ev is None:
            ev = Event(f"{frame}#{len(self.events[frame])}", frame, tup)
            self.events[frame][tup] = ev
            k = len(tup)
            fb = self.frames[frame]["bool"]
            for j, e in enumerate(tup):
                if fb and j == k - 1:
                    continue
                self.by_entity[e].add((frame, tup))
        ev.sources.append(oid)
        return ev

    def unify(self, word: str, frame: str, sigma: tuple[int, ...], support: int) -> None:
        """Map `word` onto `frame` with role order `sigma`; re-key its events so each
        fact is stored once (plan §5.1)."""
        old = self.lex[word]
        old_frame = old.frame
        assert self.words_of(old_frame) == [word], "only a word with its own frame can be unified"
        new = LexEntry(word, frame, sigma, support=support, provisional=True)
        moved = list(self.events.get(old_frame, {}).values())
        for ev in moved:
            surf = self.surface(old, ev.tup)
            tup = self.canon(new, surf)
            for oid in ev.sources:
                ev2 = self._put(frame, tup, oid)
                ev2.noise = ev2.noise or ev.noise
            for e in ev.tup:
                self.by_entity[e].discard((old_frame, ev.tup))
        if old_frame != frame and all(le.frame != old_frame for w, le in self.lex.items() if w != word):
            self.events.pop(old_frame, None)
            self.frames.pop(old_frame, None)
        self.lex[word] = new

    def split(self, word: str) -> None:
        """Undo a unification (misread fix): give the word its own frame again,
        rebuilt from its own raw observations."""
        entry = self.lex[word]
        boolean = self.frames[entry.frame]["bool"]
        # remove this word's sources from the shared frame
        oids = [i for i, (w, _, _) in enumerate(self.obs) if w == word]
        for ev in list(self.events[entry.frame].values()):
            ev.sources = [s for s in ev.sources if s not in oids]
            if not ev.sources:
                del self.events[entry.frame][ev.tup]
                for e in ev.tup:
                    self.by_entity[e].discard((entry.frame, ev.tup))
        fid = f"{self.cid}/F:{word}#split{len(self.frames)}"
        self.frames[fid] = {"arity": len(entry.sigma) - 1, "bool": boolean}
        self.lex[word] = LexEntry(word, fid, tuple(range(len(entry.sigma))), provisional=True)
        for oid in oids:
            w, args, out = self.obs[oid]
            self._put(fid, args + (out,), oid)

    # ---- frame views ------------------------------------------------------------
    def frame_data(self, frame: str) -> dict[tuple, Counter]:
        """canonical args -> Counter(out), excluding events flagged as noise."""
        d: dict[tuple, Counter] = defaultdict(Counter)
        for tup, ev in self.events.get(frame, {}).items():
            if ev.noise:
                continue
            d[tup[:-1]][tup[-1]] += len(ev.sources)
        return d

    def majority(self, frame: str) -> dict[tuple, int]:
        return {a: c.most_common(1)[0][0] for a, c in self.frame_data(frame).items()}

    def words_of(self, frame: str) -> list[str]:
        return [w for w, e in self.lex.items() if e.frame == frame]
