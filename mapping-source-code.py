import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def wczytaj_plik(filepath):
    with open(filepath, 'r') as f:
        szerokosc, wysokosc, odleglosc = map(float, f.readline().strip().split())
        macierz_wysokosci = np.loadtxt(f)
    return szerokosc, wysokosc, odleglosc, macierz_wysokosci

def oblicz_cieniowanie(macierz_wysokosci, azymut_swiatla=315, kąt_padania_swiatla=25):
    grad_y, grad_x = np.gradient(macierz_wysokosci)
    azymut_rad = np.deg2rad(azymut_swiatla)
    kat_rad = np.deg2rad(kąt_padania_swiatla)

    sx = np.sin(azymut_rad) * np.cos(kat_rad)
    sy = np.cos(azymut_rad) * np.cos(kat_rad)
    sz = np.sin(kat_rad)

    normal = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1)
    nx = grad_x / normal
    ny = grad_y / normal
    nz = 1 / normal

    illumination = nx * sx + ny * sy + nz * sz
    brightness_factor = 0.7
    illumination = (illumination - illumination.min()) / (illumination.max() - illumination.min())
    illumination = illumination * brightness_factor + (1 - brightness_factor)

    return illumination

def nakladanie_kolorow(macierz_wysokosci, illumination):
    normalized_height = ((macierz_wysokosci - np.min(macierz_wysokosci)) /
                         (np.max(macierz_wysokosci) - np.min(macierz_wysokosci)))
    rgb_image = np.zeros((macierz_wysokosci.shape[0], macierz_wysokosci.shape[1], 3))
    for i in range(macierz_wysokosci.shape[0]):
        for j in range(macierz_wysokosci.shape[1]):
            zielony = (0.5 - normalized_height[i, j]) * illumination[i, j]
            czerwony = (1 - normalized_height[i, j]) * illumination[i, j]
            rgb_image[i, j] = [czerwony, zielony, 0]
    return rgb_image

def wyswietl_teren(macierz_wysokosci):
    cieniowanie = oblicz_cieniowanie(macierz_wysokosci)
    obraz_rgb = nakladanie_kolorow(macierz_wysokosci, cieniowanie)
    plt.imshow(obraz_rgb)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Użycie: python terrain_viewer.py <plik.dem>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Plik nie istnieje: {filepath}")
        sys.exit(1)

    szerokosc, wysokosc, odleglosc, macierz_wysokosci = wczytaj_plik(filepath)
    wyswietl_teren(macierz_wysokosci)
