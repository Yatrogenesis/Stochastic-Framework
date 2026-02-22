/// LOTTO Analyzer - Multi-dimensional Mathematical Analysis Engine
/// Based on the framework from INFORME_EJECUTIVO - CHISPAZO.md
/// Author: Francisco Molina (pako.molina@gmail.com)
/// ORCID: https://orcid.org/0009-0008-6093-8267

mod models;
mod data_loader;
mod analysis;
mod ui;

fn main() -> Result<(), eframe::Error> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Starting LOTTO Analyzer v1.0.0");
    log::info!("Multi-dimensional Mathematical Analysis Engine");

    // Run the desktop application
    ui::run_app()
}
