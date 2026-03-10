mod api;
mod grpc;
mod state;

use std::net::SocketAddr;
use tokio::net::TcpListener;
use tonic::transport::Server;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use grpc::{OccupancyService, TelemetryServiceServer};
use state::create_occupancy_map;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize centralized logging pipeline
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    // 2. Instantiate thread-safe shared map
    let map = create_occupancy_map();
    
    // 3. Configure HTTP server bindings
    let rest_addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], 50051));

    info!("CRONOS Telemetry Engine initializing...");
    info!("Starting HTTP Ingestion API on {}", rest_addr);
    info!("Starting gRPC Occupancy Service on {}", grpc_addr);

    let rest_map = map.clone();
    
    // Spawn Axum ingestion server on a separate task
    let rest_task = tokio::spawn(async move {
        let app = api::create_router(rest_map);
        let listener = TcpListener::bind(rest_addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // Run gRPC server on the main execution context
    let grpc_service = OccupancyService { map };
    Server::builder()
        .add_service(TelemetryServiceServer::new(grpc_service))
        .serve(grpc_addr)
        .await?;

    rest_task.await?;

    Ok(())
}
