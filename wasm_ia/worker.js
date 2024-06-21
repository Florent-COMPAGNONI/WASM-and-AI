self.onmessage = async function (e) {
    // Import and initialize the WASM module
    const { default: init, create_mlp, train_mlp } = await import('./pkg/wasm_ia.js');
    
    // Ensure the WASM module is initialized
    await init();

    // Extract data from the message
    const { layers, X, Y, learning_rate, nb_iter, step } = e.data;

    // Create model
    model = create_mlp(layers);

    // Define the callback function to receive messages from WASM
    self.callback = (message) => {
        postMessage({ type: 'progress', data: message });
    };

    // Call the train_mlp function from the WASM module
    const updatedModel = train_mlp(model, X, Y, learning_rate, nb_iter, step);

    // Send the updated model back to the main thread
    self.postMessage({ type: 'updatedModel', data: updatedModel });
};