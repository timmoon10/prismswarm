"""Entry point: wires together particle state, fields, detector, the render
loop, and the REPL, and holds the initial scene setup."""

from __future__ import annotations

import argparse

import numpy as np

from . import fields, render, repl
from .detector import Detector
from .simulation import Simulation
from .state import ParticleState


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prismswarm — particle light simulation")
    parser.add_argument("-n", "--num-particles", type=int, default=1_000_000)
    parser.add_argument("--detector-size", type=int, default=300, help="detector resolution (square, px)")
    parser.add_argument("--display-size", type=int, default=900, help="display window size (square, px)")
    parser.add_argument("--fps", type=int, default=30, help="target frame rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wavelength", type=float, default=530.0, help="initial emission wavelength, nm")
    parser.add_argument("--exposure", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    particles = ParticleState.uniform_ball(
        n=args.num_particles, rng=rng, dim=3, radius=1.0, wavelength_nm=args.wavelength
    )

    field_catalog = {
        "radial": fields.radial_inward(speed=0.3),
        "brownian": fields.brownian(sigma=0.05),
    }

    sim = Simulation(
        state=particles,
        detector=Detector(width=args.detector_size, height=args.detector_size, half_extent=1.2),
        fields=field_catalog,
        active_field_name="radial",
        rng=rng,
        dt=1.0 / args.fps,
        exposure=args.exposure,
    )

    repl.start(namespace=dict(sim=sim, fields=fields, np=np))

    render.run(sim, display_size=(args.display_size, args.display_size), target_fps=args.fps)


if __name__ == "__main__":
    main()
