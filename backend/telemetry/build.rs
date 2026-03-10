fn main() {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["src/proto/telemetry.proto"], &["src/proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));
}
