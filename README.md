# winit-touch-test

A wasm app to show you where you are touching the screen.

Something doesn't work properly in Firefox mobile - when you
are moving two fingers at once, only one of them updates. 
This makes pinch to zoom impossible and janky.

```shell
wasm-pack build --target=web crates/touch-test-wasm
```