from astropy.cosmology import Planck18
import astropy.units as u

zs = [0.292835, 0.349605, 0.420416]
mass_bins = ["(11.0, 11.4]", "(11.4, 11.7]", "(11.7, 12.0]"]

theta = 2 * u.arcmin

for mass_bin, z in zip(mass_bins, zs):
    D_comoving = Planck18.comoving_distance(z)
    D_angular = Planck18.angular_diameter_distance(z)

    physical_scale = Planck18.kpc_proper_per_arcmin(z)
    physical_2arcmin = (physical_scale * theta).to(u.Mpc)

    print(f"logM {mass_bin}")
    print(f"  z = {z:.6f}")
    print(f"  Comoving distance         = {D_comoving:.1f}")
    print(f"  Angular diameter distance = {D_angular:.1f}")
    print(f"  Scale                     = {physical_scale:.1f}")
    print(f"  2 arcmin physical radius  = {physical_2arcmin:.3f}")
    print()
