from __future__ import annotations

import numpy as np

from encirclement3d.calibration import fit_free_decay_damping, fit_static_stiffness, measure_impact


def test_static_stiffness_fit_recovers_known_segment_parameters() -> None:
    extension = np.linspace(0.0, 0.10, 11)
    force = 42.0 * extension + 0.30
    fit = fit_static_stiffness(extension, force)
    np.testing.assert_allclose(fit.stiffness_n_per_m, 42.0, atol=1e-12)
    np.testing.assert_allclose(fit.preload_n, 0.30, atol=1e-12)
    np.testing.assert_allclose(fit.r_squared, 1.0, atol=1e-12)


def test_free_decay_fit_recovers_effective_viscous_damping() -> None:
    mass = 0.08
    damping_ratio = 0.05
    natural_frequency = 20.0
    damped_frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    time = np.arange(0.0, 2.0, 0.001)
    displacement = np.exp(-damping_ratio * natural_frequency * time) * np.cos(damped_frequency * time)
    fit = fit_free_decay_damping(time, displacement, moving_mass_kg=mass)
    np.testing.assert_allclose(fit.damping_ratio, damping_ratio, rtol=0.03)
    np.testing.assert_allclose(fit.natural_frequency_rad_s, natural_frequency, rtol=0.03)
    np.testing.assert_allclose(fit.damping_n_s_per_m, 2.0 * damping_ratio * natural_frequency * mass, rtol=0.05)


def test_impact_metrics_integrate_force_trace() -> None:
    time = np.array([0.0, 0.1, 0.2])
    force = np.array([0.0, 4.0, 0.0])
    metrics = measure_impact(time, force)
    assert metrics.peak_force_n == 4.0
    np.testing.assert_allclose(metrics.impulse_n_s, 0.4, atol=1e-12)
