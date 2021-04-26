import numpy as np
import env_randomizer_base

# Relative range.
pupper_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
pupper_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%

# Absolute range.
BATTERY_VOLTAGE_RANGE = (7.0, 8.4)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
pupper_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless

class PupperEnvRandomizer(env_randomizer_base.EnvRandomizerBase):
    def __init__(self,
                 pupper_base_mass_err_range=pupper_BASE_MASS_ERROR_RANGE,
                 pupper_leg_mass_err_range=pupper_LEG_MASS_ERROR_RANGE,
                 battery_voltage_range=BATTERY_VOLTAGE_RANGE,
                 motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
        self._pupper_base_mass_err_range = pupper_base_mass_err_range
        self._pupper_leg_mass_err_range = pupper_leg_mass_err_range
        self._battery_voltage_range = battery_voltage_range
        self._motor_viscous_damping_range = motor_viscous_damping_range

        np.random.seed(0)

