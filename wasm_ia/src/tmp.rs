use wasm_bindgen_futures::future_to_promise;
use wasm_bindgen_futures::js_sys::{Promise, Function};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
extern "C" {
    fn update_page(message: &str);
    #[wasm_bindgen(js_namespace = Promise, js_name = resolve)]
    fn resolve_promise() -> Function;
}

#[wasm_bindgen]
pub fn run_loop_async(n: u32, step: u32) -> Promise {
    future_to_promise(async move {
        for i in 0..n {
            // Simulate work
            if i % step == 0 {
                let message = format!("Step {}", i);
                update_page(&message);
                // Yield to JavaScript
                JsFuture::from(Promise::resolve(&JsValue::UNDEFINED)).await?;
            }
        }
        Ok(JsValue::UNDEFINED)
    })
}