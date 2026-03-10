fn main() {
    prost_build::Config::new()
        .compile_protos(&["src/proto/telemetry.proto"], &["src/proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));
}
