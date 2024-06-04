// Importation des modules nécessaires depuis wasm_bindgen, linreg et serde_wasm_bindgen.
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use linreg::linear_regression;
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};

pub mod mlp;

// Définition de la structure des données pour les points de données.
#[derive(Serialize, Deserialize)]
struct DataPoint {
    x: f64,
    y: f64,
}

// Définition de la structure LinearModel pour gérer le modèle de régression linéaire.
#[wasm_bindgen]
pub struct LinearModel {
    slope: f64,
    intercept: f64,
}

// Implémentation des fonctions associées à la structure LinearModel.
#[wasm_bindgen]
impl LinearModel {
    // Constructeur pour initialiser une nouvelle instance de LinearModel avec des coefficients par défaut.
    #[wasm_bindgen(constructor)]
    pub fn new() -> LinearModel {
        LinearModel {
            slope: 0.0,
            intercept: 0.0,
        }
    }

    // Fonction pour entraîner le modèle avec les données fournies.
    pub fn train(&mut self, data: JsValue) -> Result<(), JsValue> {
        // Conversion des données JavaScript en vecteur de DataPoint en utilisant from_value.
        let data_points: Vec<DataPoint> = from_value(data)?;
        // Extraction des valeurs x et y des points de données.
        let x_vals: Vec<f64> = data_points.iter().map(|point| point.x).collect();
        let y_vals: Vec<f64> = data_points.iter().map(|point| point.y).collect();

        // Calcul des coefficients de régression linéaire (pente et interception) en utilisant linear_regression.
        let (slope, intercept) = linear_regression(&x_vals, &y_vals).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.slope = slope;
        self.intercept = intercept;

        Ok(())
    }

    // Fonction pour faire une prédiction en utilisant le modèle entraîné.
    pub fn predict(&self, x: f64) -> f64 {
        // Prédiction basée sur l'équation de la ligne droite y = mx + c.
        self.slope * x + self.intercept
    }

    // Fonction pour obtenir les coefficients du modèle sous forme de valeur JavaScript.
    pub fn coefficients(&self) -> Result<JsValue, JsValue> {
        // Création d'une paire (tuple) pour les coefficients et conversion en JsValue en utilisant to_value.
        to_value(&(self.slope, self.intercept)).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
