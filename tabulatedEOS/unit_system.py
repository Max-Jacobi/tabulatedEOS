"""
This is a 1to1 copy of the unit_system.hpp used in primitive solver.
"""

from dataclasses import dataclass

def PS_SQR(x: float) -> float:
    return x * x

def PS_CUBE(x: float) -> float:
    return x * x * x

@dataclass(frozen=True)
class UnitSystem:
    # Fundamental constants
    c: float                 # Speed of light
    Gnewt: float             # Gravitational constant
    kb: float                # Boltzmann constant
    Msun: float              # Solar mass
    MeV: float               # 10^6 electronvolt

    # Base units
    length: float            # Length unit
    time: float              # Time unit
    density: float           # Number density unit
    mass: float              # Mass unit
    energy: float            # Energy unit
    pressure: float          # Pressure unit
    temperature: float       # Temperature unit
    chemicalPotential: float # Chemical potential unit

    # Conversion methods: convert values FROM this system TO another (b)
    def LengthConversion(self, b: "UnitSystem") -> float:
        return b.length / self.length

    def TimeConversion(self, b: "UnitSystem") -> float:
        return b.time / self.time

    def VelocityConversion(self, b: "UnitSystem") -> float:
        return (b.length / self.length) * (self.time / b.time)

    def DensityConversion(self, b: "UnitSystem") -> float:
        return b.density / self.density

    def MassConversion(self, b: "UnitSystem") -> float:
        return b.mass / self.mass

    def MassDensityConversion(self, b: "UnitSystem") -> float:
        return b.mass / self.mass * b.density / self.density

    def EnergyConversion(self, b: "UnitSystem") -> float:
        return b.energy / self.energy

    def EntropyConversion(self, b: "UnitSystem") -> float:
        return b.kb / self.kb

    def PressureConversion(self, b: "UnitSystem") -> float:
        return b.pressure / self.pressure

    def TemperatureConversion(self, b: "UnitSystem") -> float:
        return b.temperature / self.temperature

    def ChemicalPotentialConversion(self, b: "UnitSystem") -> float:
        return b.chemicalPotential / self.chemicalPotential


# ---- Global unit systems ----

# CGS units (CODATA 2014 consistency)
CGS = UnitSystem(
    c=2.99792458e10,                # cm/s
    Gnewt=6.67408e-8,               # cm^3 g^-1 s^-2
    kb=1.38064852e-16,              # erg K^-1
    Msun=1.98848e33,                # g
    MeV=1.6021766208e-6,            # erg

    length=1.0,                     # cm
    time=1.0,                       # s
    density=1.0,                    # g cm^-3
    mass=1.0,                       # g
    energy=1.0,                     # erg
    pressure=1.0,                   # erg / cm^3
    temperature=1.0,                # K
    chemicalPotential=1.0,          # erg
)

# Geometric units with length in kilometers
GeometricKilometer = UnitSystem(
    c=1.0,
    Gnewt=1.0,
    kb=1.0,
    Msun=CGS.Msun * CGS.Gnewt / (CGS.c * CGS.c) * 1e-5,               # km
    MeV=CGS.MeV * CGS.Gnewt / (CGS.c**4) * 1e-5,                      # km

    length=1e-5,                                                       # km
    time=CGS.c * 1e-5,                                                # km
    density=1e15,                                                     # km^-3
    mass=CGS.Gnewt / (CGS.c * CGS.c) * 1e-5,                          # km
    energy=CGS.Gnewt / (CGS.c**4) * 1e-5,                             # km
    pressure=CGS.Gnewt / (CGS.c**4) * 1e10,                           # km^-2
    temperature=CGS.kb * CGS.Gnewt / (CGS.c**4) * 1e-5,               # km
    chemicalPotential=1.0 / CGS.MeV,                                  # MeV
)

# Geometric units with length in solar masses
GeometricSolar = UnitSystem(
    c=1.0,
    Gnewt=1.0,
    kb=1.0,
    Msun=1.0,
    MeV=CGS.MeV / (CGS.c * CGS.c),                                    # MeV per Msun

    length=(CGS.c*CGS.c) / (CGS.Gnewt * CGS.Msun),                    # Msun
    time=PS_CUBE(CGS.c) / (CGS.Gnewt * CGS.Msun),                     # Msun
    density=PS_CUBE((CGS.Gnewt * CGS.Msun) / (CGS.c*CGS.c)),          # Msun^-3
    mass=1.0 / CGS.Msun,                                              # Msun
    energy=1.0 / (CGS.Msun * CGS.c * CGS.c),                          # Msun
    pressure=PS_CUBE(CGS.Gnewt / (CGS.c*CGS.c)) *                     # Msun^-2
             PS_SQR(CGS.Msun / CGS.c),
    temperature=CGS.kb / CGS.MeV,                                     # MeV
    chemicalPotential=1.0 / CGS.MeV,                                  # MeV
)

# Nuclear units
Nuclear = UnitSystem(
    c=1.0,
    Gnewt=CGS.Gnewt * CGS.MeV / (CGS.c**4) * 1e13,                    # fm
    kb=1.0,
    Msun=CGS.Msun * (CGS.c*CGS.c) / CGS.MeV,                          # MeV
    MeV=1.0,

    length=1e13,                                                      # fm
    time=CGS.c * 1e13,                                                # fm
    density=1e-39,                                                    # fm^-3
    mass=(CGS.c*CGS.c) / CGS.MeV,                                     # MeV
    energy=1.0 / CGS.MeV,                                             # MeV
    pressure=1e-39 / CGS.MeV,                                         # MeV / fm^3
    temperature=CGS.kb / CGS.MeV,                                     # MeV
    chemicalPotential=1.0 / CGS.MeV,                                  # MeV
)

unit_systems = {
    "CGS": CGS,
    "GeometricSolar": GeometricSolar,
    "GeometricKilometer": GeometricKilometer,
    "Nuclear": Nuclear,
    }
