"""MAP-B hypervectors (plan §4.1). Fixed random codes; nothing here is learned."""
from __future__ import annotations

import hashlib

import numpy as np

D = 10_000


class VSA:
    def __init__(self, dim: int = D, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}

    def hv(self, name: str) -> np.ndarray:
        """Atomic bipolar vector, deterministic per (seed, name)."""
        v = self._cache.get(name)
        if v is None:
            h = int.from_bytes(hashlib.sha256(f"{self.seed}:{name}".encode()).digest()[:8], "little")
            v = np.random.default_rng(h).choice(np.array([-1, 1], dtype=np.int8), self.dim)
            self._cache[name] = v
        return v

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).astype(np.int8)

    def bundle(self, vs: list[np.ndarray], tie_name: str) -> np.ndarray:
        """Thresholded bundle; ties broken by a per-bundle seeded vector (plan §4.1 warning)."""
        s = np.sum(np.stack([v.astype(np.int32) for v in vs]), axis=0)
        out = np.sign(s).astype(np.int8)
        z = out == 0
        if z.any():
            out[z] = self.hv("tie:" + tie_name)[z]
        return out

    @staticmethod
    def key_sum(vs: list[np.ndarray]) -> np.ndarray:
        """Un-thresholded integer bundle, used for router keys (plan §4.3)."""
        return np.sum(np.stack([v.astype(np.int32) for v in vs]), axis=0)

    @staticmethod
    def sim(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))

    def frame_vector(self, frame_id: str, bindings: dict[str, str]) -> np.ndarray:
        """F = [sum_r ROLE_r (x) filler_r]."""
        parts = [self.bind(self.hv("role:" + r), self.hv("ent:" + f)) for r, f in sorted(bindings.items())]
        return self.bundle(parts, "frame:" + frame_id + ":" + ",".join(f"{r}={f}" for r, f in sorted(bindings.items())))

    def unbind_role(self, F: np.ndarray, role: str) -> np.ndarray:
        return self.bind(self.hv("role:" + role), F)

    def cleanup(self, v: np.ndarray, candidates: list[str]) -> tuple[str, float]:
        """Nearest item in an item memory of entity names."""
        M = np.stack([self.hv("ent:" + c) for c in candidates]).astype(np.int32)
        s = M @ v.astype(np.int32) / self.dim
        i = int(np.argmax(s))
        return candidates[i], float(s[i])

    def resonator(self, s: np.ndarray, codebooks: list[list[str]], iters: int = 200):
        """Factor s = x1 (x) x2 (x) ... into one item per codebook (Frady et al. 2020).
        Returns (names, converged)."""
        books = [np.stack([self.hv(n) for n in cb]).astype(np.int32) for cb in codebooks]
        est = [np.sign(B.sum(axis=0)).astype(np.int32) for B in books]
        for e in est:
            e[e == 0] = 1
        for _ in range(iters):
            changed = False
            for i, B in enumerate(books):
                other = np.ones(self.dim, dtype=np.int32)
                for j, e in enumerate(est):
                    if j != i:
                        other *= e
                new = np.sign(B.T @ (B @ (s.astype(np.int32) * other)))
                new[new == 0] = 1
                if not np.array_equal(new, est[i]):
                    changed = True
                est[i] = new.astype(np.int32)
            if not changed:
                break
        names = []
        for i, B in enumerate(books):
            names.append(codebooks[i][int(np.argmax(B @ est[i]))])
        return names, not changed
