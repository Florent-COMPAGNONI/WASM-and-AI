use rand::Rng;
use std::io::{Write, BufReader, BufRead, Error};
use std::fs::File;
use chrono::{Datelike, DateTime, Local, Timelike, Utc};

// tmp
use std::any::type_name;



fn type_of<T>(_: T) -> &'static str { 
    type_name::<T>()
}


#[repr(C)]
#[derive(Debug)]
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



#[no_mangle]
fn create_mlp(layers: &Vec<i32>) -> MLP {

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

    return model;
}


// TODO generic type ?
fn array_to_matrix(array: &[f64], sub_array_length: usize) -> Vec<Vec<f64>> {

    let rows = array.len() / sub_array_length as usize;
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.; sub_array_length]; rows];

    for i in 0..rows {
        for j in 0..sub_array_length {
            let index = i * sub_array_length  + j;
            matrix[i][j] = array[index];
        }
    }
    return matrix;
}


fn predict_mlp(model: &mut MLP, sample_input: &Vec<f64>, is_classification: i32) -> Vec<f64>{

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
    return model.inputs[(model.nb_layers) as usize][1..].to_vec();
}


#[no_mangle]
pub extern "C" fn wrap_predict(model_ptr: *mut MLP, input_ptr: *const f64, input_length: i32, is_classification: i32) -> *mut f64 {

    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    let inputs: &[f64] = unsafe {
        std::slice::from_raw_parts(input_ptr, input_length as usize)
    };

    let prediction: Vec<f64> = predict_mlp(model, &inputs.to_vec(), is_classification);

    return prediction.leak().as_mut_ptr();
}


// TODO output is a matrix ...
#[no_mangle]
pub extern "C" fn train_mlp(model_ptr: *mut MLP,
                            inputs_ptr: *const f64, inputs_length: i32, inputs_sublength: i32,
                            expected_outputs_ptr: *const f64, expected_outputs_length: i32, expected_outputs_sublength: i32,
                            is_classification: i32, learning_rate: f64, nb_iter: i32, epoch: i32)
{

    // convert from raw part
    // TODO match Some / None
    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };
    model.learning_rate = learning_rate;
    model.nb_iter = nb_iter;

    let tmp_inputs: &[f64] = unsafe {
        std::slice::from_raw_parts(inputs_ptr, inputs_length as usize)
    };
    let inputs: Vec<Vec<f64>> = array_to_matrix(tmp_inputs, inputs_sublength as usize);

    let tmp_excepted_outputs: &[f64] = unsafe {
        std::slice::from_raw_parts(expected_outputs_ptr, expected_outputs_length as usize)
    };
    // TODO expected output can be a matrix ???!!!
    let excepted_outputs: Vec<Vec<f64>> = array_to_matrix(tmp_excepted_outputs, expected_outputs_sublength as usize);

    // reset logs
    model.logs = Vec::new();

    // let start_time = Utc::now().time();

    for i in 0..model.nb_iter {

        // random choice of an input, and of the expected output linked to it
        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0..inputs.len());

        let sample_input: &Vec<f64> = &inputs[random_index];
        let sample_expected_output: &Vec<f64> = &excepted_outputs[random_index];

        // update the inputs of each layers via predict func + store current iter error
        let current_error: Vec<f64> =  predict_mlp(model, sample_input, is_classification);

        model.logs.push(current_error);
        model.logs.push((*sample_expected_output.clone()).to_owned());


        // calculate semi gradient for the last layers
        for i in 1..(model.layers[(model.nb_layers) as usize] + 1) as usize {
            let mut semi_gradient: f64 = model.inputs[(model.nb_layers) as usize][i] - sample_expected_output[i - 1];

            if is_classification == 1 {
                semi_gradient *= 1. - model.inputs[(model.nb_layers) as usize][i].powi(2);
            }
            model.deltas[(model.nb_layers) as usize][i] = semi_gradient;
        }

        // backpropagation of semi gradient (from before last to second layer)
        for l in (1..(model.nb_layers + 1) as usize).rev() {
            for i in 1..(model.layers[l - 1] + 1) as usize {

                // weighted sum calculation
                let mut total: f64 = 0.;
                for j in 1..(model.layers[l] + 1) as usize {
                    total += model.weights[l][i][j] * model.deltas[l][j];
                }

                let semi_gradient = total * (1. - model.inputs[l - 1][i].powi(2));
                model.deltas[l - 1][i] = semi_gradient;
            }
        }

        // update weights using learning rate, inputs and semi_gradients
        for l in 1..(model.nb_layers + 1) as usize {
            for i in 0..(model.layers[l - 1] + 1) as usize {
                for j in 0..(model.layers[l] + 1) as usize {
                    model.weights[l][i][j] -= model.learning_rate * model.inputs[l - 1][i] * model.deltas[l][j];
                }
            }
        }
        println!("epoch : {:05} - iter: {:07}", epoch, i);
    }
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


#[no_mangle]
pub extern "C" fn wrap_serialize_logs(model_ptr: *mut MLP, epoch: i32) {

    // TODO match Some / None
    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    serialize_logs(model, epoch);
}


fn format_model_infos(model: &mut MLP) -> String {

    let mut infos: String = "layers;".to_owned();

    for layer in &model.layers {
        infos.push_str(&format!("{};", layer));
    }
    infos.push_str(&format!("learning_rate;{};", model.learning_rate));
    infos.push_str(&format!("nb_iter;{};\n", model.nb_iter));

    return infos;
}


#[no_mangle]
pub extern "C" fn wrap_serialize_model(model_ptr: *mut MLP) {

    // TODO match Some / None
    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    serialize_model(model);
}


fn serialize_model(model: &mut MLP) {

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

    let mut model_infos: String = format_model_infos(model);

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


#[no_mangle]
pub extern "C" fn free_mlp(model: *mut MLP) {
    unsafe {
        Box::from_raw(model);
    }
}


// TODO test
#[no_mangle]
pub extern "C" fn free_f64_array(tab: *mut f64, tab_size: i32) {
    unsafe {
        Vec::from_raw_parts(tab, tab_size as usize, tab_size as usize);
    }
}




// tmp

#[no_mangle]
pub extern "C" fn display_mlp_raw_ptr(model: *const MLP) {

    // TODO match Some / None
    let model_safe = unsafe{ &*model.as_ref().unwrap() };
    display_mlp(model_safe);
}


fn display_mlp(model: &MLP) {

    println!("nb of layer: {}\n", model.nb_layers);

    println!("layers : \n{:?}\n", model.layers);

    println!("weights: \n{:?}\n", model.weights);

    // println!("deltas: \n{:?}\n", model.deltas);

    println!("inputs: \n{:?}\n", model.inputs);

    // println!("logs: \n{:?}\n", model.logs);
}


#[no_mangle]
pub extern "C" fn test_i32_to_f64(x: f64) {

    println!("{}", x);
}


#[no_mangle]
pub extern "C" fn test_array_return() -> *mut i32 {

    let arr = vec![1,2,3,4];

    return arr.leak().as_mut_ptr();
}

#[no_mangle]
pub extern "C" fn test_matrix_return() -> *mut *mut i32 {

    let arr = vec![
        vec![0,0],
        vec![1,1],
    ];

    let mut result: Vec<*mut i32> = Vec::with_capacity(2);

    for i in arr {
        result.push(i.leak().as_mut_ptr());
    }

    return result.leak().as_mut_ptr();
}



#[cfg(test)]
mod tests {
    use crate::mlp::create_mlp;
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

    #[test]
    fn test_mlp() {
        let layers = vec![2, 3, 1];
        let mut model = create_mlp(&layers);

        let inputs = vec![
            vec![0., 0.],
            vec![0., 1.],
            vec![1., 0.],
            vec![1., 1.],
        ];

        let expected_outputs = vec![
            vec![0.],
            vec![1.],
            vec![1.],
            vec![0.],
        ];

    }
}
