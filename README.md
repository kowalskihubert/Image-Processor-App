# Image-Processor-App - Narzędzia do Przetwarzania Obrazów

## 📌 Opis Projektu
Aplikacja desktopowa w języku Python implementująca podstawowe i zaawansowane operacje przetwarzania obrazów, w tym:

- **Transformacje obrazu**: 
  - Skala szarości (Average, Luminosity, Custom)
  - Negatyw
  - Korekcja jasności i kontrastu
  - Transformacja gamma
  - Binaryzacja z progowanie

- **Filtry przestrzenne**:
  - Rozmycie Gaussa
  - Filtr uśredniający
  - Wyostrzanie
  - Własne jądra splotowe 3x3

- **Detekcja krawędzi**:
  - Operator Sobela
  - Krzyż Robertsa
  - Algorytm Canny'ego

- **Analiza statystyczna**:
  - Histogramy (RGB/grayscale)
  - Projekcje poziome/pionowe
  - Wykrywanie koloru dominującego

## 🛠 Wymagania
- Python 3.8+
- NumPy
- PyQt5
- scikit-learn (dla detekcji koloru dominującego)
- Matplotlib (dla wizualizacji statystyk)

## 📘 Dokumentacja
[Pełna dokumentacja PDF](Documentation.pdf)


## 🚀 Szybki Start

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/kowalskihubert/Image-Processor-App.git
   cd Image-Processor-App
   ```
   
2. **Utwórz i aktywuj środowisko wirtualne:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    venv\Scripts\activate     # Windows
    ```

3. **Zainstaluj wymagane biblioteki:**

    ```bash
    pip install -r requirements.txt
    ```
    
4. Uruchom aplikację:

    ```bash
    python app.py
    ```

