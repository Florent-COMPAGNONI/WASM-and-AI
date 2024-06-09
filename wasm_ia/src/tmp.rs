use wasm_bindgen_futures::future_to_promise;
use wasm_bindgen_futures::js_sys::{Promise, Function};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
extern "C" {
    fn update_page(message: &str);
    #[wasm_bindgen(js_namespace = window, js_name = setTimeout)]
    fn set_timeout(f: &Function, delay: u32) -> i32;
}

#[wasm_bindgen]
pub fn run_loop(n: u32, step: u32) {
    let closure = Closure::wrap(Box::new(move || {
        run_loop_step(0, n, step);
    }) as Box<dyn Fn()>);

    set_timeout(closure.as_ref().unchecked_ref(), 0);
    closure.forget();
}

fn run_loop_step(current: u32, n: u32, step: u32) {
    if current >= n {
        return;
    }

    if current % step == 0 {
        let message = format!("Step {}", current);
        update_page(&message);
    }

    let next_closure = Closure::wrap(Box::new(move || {
        run_loop_step(current + 1, n, step);
    }) as Box<dyn Fn()>);

    set_timeout(next_closure.as_ref().unchecked_ref(), 0);
    next_closure.forget();
}