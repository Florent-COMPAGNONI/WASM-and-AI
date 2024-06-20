use rand::Rng;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen_futures::future_to_promise;
use wasm_bindgen_futures::js_sys::Promise;
use std::time::Duration;
use async_std::task;


#[wasm_bindgen]
extern {
    // external JS function
    fn update_page(message: &str);

    #[wasm_bindgen(js_namespace = Object)]
    fn assign(target: &JsValue, source: &JsValue);
}


#[wasm_bindgen]
#[derive(Debug, Serialize, Deserialize)]
pub struct MLP {
    layers: Vec<i32>,
    nb_layers: i32,
    weights: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
    learning_rate: f64,
    nb_iter: i32,
}


fn init_weights(layers: &Vec<i32>) -> Vec<Vec<Vec<f64>>> {

    let mut rng = rand::thread_rng();
    let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();

    for l in 0..layers.len() {

        // empty for the 1st layer
        if l == 0 {
            weights.push(Vec::new());
            continue;
        }

        let rows: usize = (layers[l - 1] + 1) as usize;
        let columns: usize = (layers[l] + 1) as usize;

        // init
        let mut layer_weights: Vec<Vec<f64>> = vec![vec![0.; columns]; rows];

        // for each neurons of each hidden layer
        for i  in 0..(layers[l-1] + 1) as usize{
            for j in 0..(layers[l] + 1) as usize {
                // one empty (the bias unit) the other random
                layer_weights[i][j] = if j == 0 { 0. } else { rng.gen_range(-1.0..1.0) };
            }
        }
        weights.push(layer_weights);
    }
    return weights;
}


#[wasm_bindgen]
pub fn create_mlp(layers_js: JsValue) -> JsValue {

    let layers: Vec<i32> = from_value(layers_js).unwrap();
    // init weights randomly
    let weights: Vec<Vec<Vec<f64>>> = init_weights(&layers);

    // allocation of deltas and inputs
    let mut deltas: Vec<Vec<f64>> = Vec::new();
    let mut inputs: Vec<Vec<f64>> = Vec::new();

    for i in 0..layers.len() {

        let layer_deltas: Vec::<f64> = vec![0.; (layers[i] + 1) as usize];

        let mut layer_inputs: Vec::<f64> = vec![0.; (layers[i] + 1) as usize];
        // the bias neuron
        layer_inputs[0] = 1.;

        deltas.push(layer_deltas);
        inputs.push(layer_inputs);
    }

    let model:MLP = MLP {
        layers: layers.clone(),
        nb_layers: (layers.len() -1) as i32,
        weights,
        deltas,
        inputs,
        learning_rate: 0.,
        nb_iter: 0,
    };

    return to_value(&model).unwrap();
}


#[wasm_bindgen]
pub fn predict_mlp(model_js: JsValue, sample_input_js: JsValue) -> JsValue{

    let mut model: MLP = from_value(model_js).unwrap();
    let sample_input: Vec<f64> = from_value(sample_input_js).unwrap();

    let prediction: Vec<f64> = predict_mlp_internal(&mut model, &sample_input);

    return to_value(&prediction).unwrap()
}


pub fn predict_mlp_internal(model: &mut MLP, sample_input: &Vec<f64>) -> Vec<f64>{

    // entry layer = input
    for i in 0..model.layers[0] as usize {
        model.inputs[0][i+1] = sample_input[i];
    }

    // we calculate the weighted sum to have the inputs on each hidden layers
    for l in 1..(model.nb_layers + 1) as usize {

        for j in 1..(model.layers[l] + 1) as usize {
            let mut total: f64 = 0.;
            for i in 0..(model.layers[l - 1] + 1) as usize {
                total += model.weights[l][i][j] * model.inputs[l-1][i];
            }
            // use hyperbolic tangent
            total = total.tanh();

            model.inputs[l][j] = total;
        }
    }
    return model.inputs[(model.nb_layers) as usize][1..].to_vec()
}


#[wasm_bindgen]
pub fn train_mlp(model_js: JsValue, inputs_js: JsValue, expected_outputs_js: JsValue, learning_rate: f64, nb_iter: i32, step: i32) -> Promise {
    future_to_promise(async move {
        let mut model: MLP = from_value(model_js.clone()).map_err(|e| JsValue::from_str(&format!("Error parsing model: {:?}", e)))?;
        let inputs: Vec<Vec<f64>> = from_value(inputs_js).map_err(|e| JsValue::from_str(&format!("Error parsing inputs: {:?}", e)))?;
        let expected_outputs: Vec<Vec<f64>> = from_value(expected_outputs_js).map_err(|e| JsValue::from_str(&format!("Error parsing expected outputs: {:?}", e)))?;

        model.learning_rate = learning_rate;
        model.nb_iter = nb_iter;

        let mut errors: Vec<f64> = Vec::<f64>::new();

        for current in 0..model.nb_iter {
            let mut rng = rand::thread_rng();
            let random_index = rng.gen_range(0..inputs.len());

            let sample_input: &Vec<f64> = &inputs[random_index];
            let sample_expected_output: &Vec<f64> = &expected_outputs[random_index];

            let current_output: Vec<f64> = predict_mlp_internal(&mut model, sample_input);

            let mut current_error: Vec<f64> = vec![0.0; sample_expected_output.len()];
            for (j, &output) in current_output.iter().enumerate() {
                current_error[j] = output - sample_expected_output[j];
            }


            for j in 1..(model.layers[(model.nb_layers) as usize] + 1) as usize {
                let mut semi_gradient: f64 = model.inputs[(model.nb_layers) as usize][j] - sample_expected_output[j - 1];

                semi_gradient *= 1. - model.inputs[(model.nb_layers) as usize][j].powi(2);

                model.deltas[(model.nb_layers) as usize][j] = semi_gradient;
            }

            for l in (1..(model.nb_layers + 1) as usize).rev() {
                for i in 1..(model.layers[l - 1] + 1) as usize {
                    let mut total: f64 = 0.;
                    for j in 1..(model.layers[l] + 1) as usize {
                        total += model.weights[l][i][j] * model.deltas[l][j];
                    }

                    let semi_gradient = total * (1. - model.inputs[l - 1][i].powi(2));
                    model.deltas[l - 1][i] = semi_gradient;
                }
            }

            for l in 1..(model.nb_layers + 1) as usize {
                for i in 0..(model.layers[l - 1] + 1) as usize {
                    for j in 0..(model.layers[l] + 1) as usize {
                        model.weights[l][i][j] -= model.learning_rate * model.inputs[l - 1][i] * model.deltas[l][j];
                    }
                }
            }

            errors.extend_from_slice(&current_error);

            if (current + 1) % step == 0 || current == 0 {
                let mse = calculate_mse(&errors);
                let message = format!("{}:{:.6}", current+1, mse);

                update_page(&message);
                errors.clear();

                task::sleep(Duration::from_nanos(1)).await;
            }
        }
        // update js model
        let updated_model_js = to_value(&model).map_err(|e| JsValue::from_str(&format!("Error serializing model: {:?}", e)))?;
        assign(&model_js, &updated_model_js);
        Ok(JsValue::UNDEFINED)
    })
}


fn calculate_mse(errors: &[f64]) -> f64 {
    let sum_of_squares: f64 = errors.iter().map(|&e| e.powi(2)).sum();
    sum_of_squares / errors.len() as f64
}
