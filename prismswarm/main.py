"""Entry point: wires together particle state, fields, detector, the render
loop, and the REPL, and holds the initial scene setup."""

from __future__ import annotations

import argparse

import numpy as np

from . import fields, gains, render, repl, spectra
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
    parser.add_argument(
        "--spectrum", choices=["blackbody", "mono"], default="blackbody", help="initial emission spectrum"
    )
    parser.add_argument(
        "--temperature", type=float, default=5778.0, help="blackbody temperature, K (--spectrum blackbody)"
    )
    parser.add_argument("--wavelength", type=float, default=530.0, help="emission wavelength, nm (--spectrum mono)")
    parser.add_argument("--exposure", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.spectrum == "blackbody":
        spectrum = spectra.blackbody(args.temperature)
    else:
        spectrum = spectra.monochrome(args.wavelength)

    particles = ParticleState.gaussian(n=args.num_particles, rng=rng, spectrum=spectrum, dim=3, sigma=0.3)

    field_catalog = {
        "radial": fields.radial_field(profile=fields.constant(-0.3)),
        "white_noise": fields.white_noise_field(sigma=0.05),
        "rotational": fields.tangential_field(profile=fields.linear(1.0)),
        "confining": fields.radial_field(profile=fields.exponential_ramp(rate=1.0, amplitude=-0.3)),
        "sinusoidal": fields.axial_field(profile=fields.sinusoidal(), rng=rng),
    }

    sim = Simulation(
        state=particles,
        detector=Detector(width=args.detector_size, height=args.detector_size, half_extent=1.2),
        fields=field_catalog,
        active_field_name="white_noise",
        rng=rng,
        spectrum=spectrum,
        dt=1.0 / args.fps,
        exposure=args.exposure,
    )

    repl.start(namespace=dict(sim=sim, fields=fields, gains=gains, spectra=spectra, np=np))

    render.run(sim, display_size=(args.display_size, args.display_size), target_fps=args.fps)


if __name__ == "__main__":
    main()
