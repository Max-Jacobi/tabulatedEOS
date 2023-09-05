from collections import namedtuple

Units = {"Rho": 6.175828477586656e+17,  # g/cm^3
         "Eps":  8.9875517873681764e+20,  # erg/g
         "Press":  5.550725674743868e+38,  # erg/cm^3
         "Mass": 1.988409870967742e+33,  # g
         "Energy": 1.7870936689836656e+3,  # 50 erg
         "Time": 0.004925490948309319,  # ms
         "Length":  1.4766250382504018}  # km
RUnits = {"Rho":  1.6192159539877191e-18,
          "Press":  1.8015662430410847e-39,
          "Eps":  1.1126500560536184e-21,
          "Mass":  5.028992139685286e-34,
          "Energy":  5.595508386114039e-55,
          "Time":  203.02544670054692,
          "Length":  0.6772199943086858}


# create a class from the dictionary of units for easy access
_lower_U = {k.lower(): v for k, v in Units.items()}
_lower_RU = {k.lower(): v for k, v in RUnits.items()}
U = namedtuple('Units', _lower_U.keys())(**_lower_U)
RU = namedtuple('RUnits', _lower_RU.keys())(**_lower_RU)
