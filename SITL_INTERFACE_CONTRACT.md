# PX4 SITL Capture Interface Contract

## Status

This repository currently has no PX4, Gazebo, or MAVSDK runtime installed.
The code in `src/encirclement3d/sitl_contract.py` is therefore a tested,
transport-independent contract, not a PX4 SITL result.

## Inputs to the High-Level Supervisor

Every control cycle must provide a common-clock `CaptureSITLObservation`:

- Four defender positions and velocities in metres and metres per second.
- A timestamped target position, velocity, and 3x3 position covariance.
- Obstacles as `[x, y, radius, height]` rows in metres.
- Net mode, deployment progress, minimum mesh margin, peak tension, and peak
  strain from the capture mechanism.
- Communication age measured at the decision endpoint.

## Output to PX4 Adapter

The only high-level output is a timestamped `CaptureSITLCommand`: four bounded
world-frame velocity setpoints and one explicit `hold`, `deploy`, `reel`, or
`abort` command. A future MAVSDK/MAVLink adapter must reject expired commands,
log receive/send/acknowledge timestamps, and implement Offboard loss and
geofence fallbacks outside this module.

## Required Gate

`CaptureSITLCommandGate` rejects non-abort commands when any of these apply:

1. Defender velocity or command horizon exceeds its limit.
2. Target estimate, net telemetry, or communication age is stale.
3. Target covariance exceeds the declared uncertainty bound.
4. Net telemetry reports fault, tension, or strain beyond its calibrated limit.
5. Deploy/reel command conflicts with the reported net mode.

Only an explicitly bounded `abort` is accepted during stale sensing or net
overload. The PX4 adapter must map that command to the separately specified
safe-abort behavior; accepting it in the contract does not command motors.
