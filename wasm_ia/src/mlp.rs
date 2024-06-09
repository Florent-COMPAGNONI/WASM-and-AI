use rand::Rng;
use std::io::Write;
use std::fs::File;
use chrono::{Datelike, DateTime, Local, Timelike};
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen_futures::future_to_promise;
use wasm_bindgen_futures::js_sys::{Function, Promise};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen::prelude::*;

use wasm_bindgen::JsCast;
use web_sys::window;
use std::cell::RefCell;
use std::rc::Rc;

#[wasm_bindgen]
extern "C" {
    /* This code declare JS function to be called from Rust */
    fn update_loss(epoch: i32, iter: i32, loss: f64);
    fn set_timeout(f: &Function, delay: u32) -> i32;
}


#[wasm_bindgen]
#[derive(Debug, Serialize, Deserialize)]
pub struct MLP {
    layers: Vec<i32>,
    nb_layers: i32,
    weights: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
    logs: Vec<Vec<f64>>,
    learning_rate: f64,
    nb_iter: i32,
}


struct TrainState {
    model: MLP,
    inputs: Vec<Vec<f64>>,
    expected_outputs: Vec<Vec<f64>>,
    is_classification: i32,
    learning_rate: f64,
    nb_iter: i32,
    epoch: i32,
    current_iter: i32,
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
        logs: Vec::new(),
        learning_rate: 0.,
        nb_iter: 0,
    };

    return to_value(&model).unwrap();
}


#[wasm_bindgen]
pub fn predict_mlp(model_js: JsValue, sample_input_js: JsValue, is_classification: i32) -> JsValue{

    let mut model: MLP = from_value(model_js).unwrap();
    let sample_input: Vec<f64> = from_value(sample_input_js).unwrap();

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
            // use hyperbolic tangent excepted for the output of regression case
            if l < model.nb_layers as usize || is_classification == 1 {
                total = total.tanh();
            }
            model.inputs[l][j] = total;
        }
    }
    return to_value(&model.inputs[(model.nb_layers) as usize][1..].to_vec()).unwrap()
}


// TODO output is a matrix ...
#[wasm_bindgen]
pub fn train_mlp(model_js: JsValue, inputs_js: JsValue, expected_outputs_js: JsValue, is_classification: i32, learning_rate: f64, nb_iter: i32, epoch: i32)
{
    let mut model: MLP = from_value(model_js).unwrap();
    let inputs: Vec<Vec<f64>> = from_value(inputs_js).unwrap();
    let expected_outputs: Vec<Vec<f64>> = from_value(expected_outputs_js).unwrap();

    // convert from raw part
    model.learning_rate = learning_rate;
    model.nb_iter = nb_iter;

    // reset logs
    model.logs = Vec::new();

    // let start_time = Utc::now().time();

    let state = Rc::new(RefCell::new(TrainState {
        model,
        inputs,
        expected_outputs,
        is_classification,
        learning_rate,
        nb_iter,
        epoch,
        current_iter: 0,
    }));

    let closure: Rc<RefCell<Option<Closure<dyn Fn()>>>> = Rc::new(RefCell::new(None));
    let closure_clone = closure.clone();

    *closure.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut state = state.borrow_mut();


        if state.current_iter >= state.nb_iter {
            return;
        }

        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0..state.inputs.len());

        let sample_input = &state.inputs[random_index];
        let sample_expected_output = &state.expected_outputs[random_index];

        let current_error: Vec<f64> = from_value(predict_mlp(to_value(&state.model).unwrap(), to_value(sample_input).unwrap(), state.is_classification)).unwrap();

        state.model.logs.push(current_error.clone());
        state.model.logs.push(sample_expected_output.clone());

        for i in 1..(state.model.layers[(state.model.nb_layers) as usize] + 1) as usize {
            let mut semi_gradient: f64 = state.model.inputs[(state.model.nb_layers) as usize][i] - sample_expected_output[i - 1];

            if state.is_classification == 1 {
                semi_gradient *= 1. - state.model.inputs[(state.model.nb_layers) as usize][i].powi(2);
            }
            state.model.deltas[(state.model.nb_layers) as usize][i] = semi_gradient;
        }

        for l in (1..(state.model.nb_layers + 1) as usize).rev() {
            for i in 1..(state.model.layers[l - 1] + 1) as usize {
                let mut total: f64 = 0.;
                for j in 1..(state.model.layers[l] + 1) as usize {
                    total += state.model.weights[l][i][j] * state.model.deltas[l][j];
                }

                let semi_gradient = total * (1. - state.model.inputs[l - 1][i].powi(2));
                state.model.deltas[l - 1][i] = semi_gradient;
            }
        }

        for l in 1..(state.model.nb_layers + 1) as usize {
            for i in 0..(state.model.layers[l - 1] + 1) as usize {
                for j in 0..(state.model.layers[l] + 1) as usize {
                    state.model.weights[l][i][j] -= state.model.learning_rate * state.model.inputs[l - 1][i] * state.model.deltas[l][j];
                }
            }
        }

        if state.current_iter % state.epoch == 0 {
            let loss: f64 = current_error.iter().sum::<f64>() / current_error.len() as f64;
            update_loss(state.epoch, state.current_iter, loss);
        }

        state.current_iter += 1;

        set_timeout(closure_clone.borrow().as_ref().unwrap().as_ref().unchecked_ref(), 0);
    }) as Box<dyn Fn()>));

    set_timeout(closure.borrow().as_ref().unwrap().as_ref().unchecked_ref(), 0);
    closure.borrow().as_ref().unwrap().forget();
}


fn serialize_logs(model: &mut MLP, epoch: i32) {

    let path: String = format!("train_loss_epoch_{}.csv", epoch);

    let mut output: File = File::create(path).expect("failed to create file");

    for log in &model.logs {
        for val in log {
            write!(output, "{};", format!("{:?}", val)).expect("failed to write");
        }
        writeln!(output, "").expect("failed to write");
    }
}


fn format_model_infos(model: &MLP) -> String {

    let mut infos: String = "layers;".to_owned();

    for layer in &model.layers {
        infos.push_str(&format!("{};", layer));
    }
    infos.push_str(&format!("learning_rate;{};", model.learning_rate));
    infos.push_str(&format!("nb_iter;{};\n", model.nb_iter));

    return infos;
}


fn serialize_model(model: &MLP) {

    let now: DateTime<Local> = Local::now();

    let path: String = format!(
        "model_{:02}_{:02}_{:04}_{:02}h{:02}.csv",
        now.day(),
        now.month(),
        now.year(),
        now.hour(),
        now.minute(),
    );

    let mut output: File = File::create(path).expect("failed to create file");

    let model_infos: String = format_model_infos(model);

    write!(output, "{}", model_infos).expect("failed to write");

    for layer in &model.weights {
        for neurons in layer {
            for weight in neurons {
                write!(output, "{};", weight).expect("failed to write");
            }
            writeln!(output, "").expect("failed to write");
        }
    }
}


fn display_mlp(model: &MLP) {

    println!("nb of layer: {}\n", model.nb_layers);

    println!("layers : \n{:?}\n", model.layers);

    println!("weights: \n{:?}\n", model.weights);

    // println!("deltas: \n{:?}\n", model.deltas);

    println!("inputs: \n{:?}\n", model.inputs);

    // println!("logs: \n{:?}\n", model.logs);
}





#[cfg(test)]
mod tests {
    use crate::mlp::{create_mlp, predict_mlp, train_mlp};

    // use std::collections::HashMap;
    // use chrono::{Datelike, Local, Timelike, Utc};

    // #[test]
    // fn it_works() {
    //     assert_eq!(2+2, 4);
    // }


    // #[test]
    // fn slice_to_matrix() {

    //     let a = vec![1, 2, 3];
    //     let b = vec![4, 5, 6];
    //     let c = vec![7, 8, 9];
    //     let mut a2: Vec<Vec<i32>> = Vec::new();
    //     a2.push(a);
    //     a2.push(b);
    //     a2.push(c);

    //     let a2_bis = &[1.,2.,3.,4.,5.,6.,7.,8.,9.];

    //     // assert_eq!(a2, array_to_matrix(a2_bis, 3));
    // }

    // #[test]
    // fn test_hashmap() {
    //     let mut logs: HashMap<i32, f64> = HashMap::<i32, f64>::new();

    //     logs.insert(1, 5.);
    //     logs.insert(2, 5.);
    //     logs.insert(3, 5.);
    //     logs.insert(4, 5.);
    //     logs.insert(5, 5.);
    //     logs.insert(6, 5.);
    //     logs.insert(1, 5.);

    //     dbg!(logs);
    // }


    // #[test]
    // fn test_file_name() {
    //     let now = Local::now();

    //     // _lr_{}_ly_{}_{}
    //     println!("{:02}_{:02}_{:04}_{:02}h{:02}_{}",
    //         now.day(),
    //         now.month(),
    //         now.year(),
    //         now.hour(),
    //         now.minute(),
    //         0.2
    //     );
    // }

    // #[test]
    // fn test_mlp() {
    //     let layers = vec![2, 3, 1];
    //     let mut model = create_mlp(&layers);

    //     let inputs = vec![
    //         vec![0., 0.],
    //         vec![0., 1.],
    //         vec![1., 0.],
    //         vec![1., 1.],
    //     ];

    //     let expected_outputs = vec![
    //         vec![0.],
    //         vec![1.],
    //         vec![1.],
    //         vec![0.],
    //     ];

    //     dbg!("{:?}",predict_mlp(&mut model, &inputs[0], 1)); 
    //     train_mlp(&mut model, &inputs, &expected_outputs, 1, 0.1, 1000, 0);

    // }
}
