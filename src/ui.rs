/// Desktop UI module using egui/eframe
use crate::analysis::LottoAnalyzer;
use crate::data_loader::DataLoader;
use crate::models::{CompleteAnalysisResult, ConfidenceLevel, LotteryDataset};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

#[derive(Default)]
pub struct LottoApp {
    /// Current loaded dataset
    dataset: Option<LotteryDataset>,

    /// Analysis result
    analysis_result: Option<CompleteAnalysisResult>,

    /// Selected confidence level
    confidence_level: ConfidenceLevel,

    /// Status message
    status_message: String,

    /// Analysis in progress
    analyzing: bool,

    /// Channel for receiving analysis results
    result_receiver: Option<Receiver<Result<CompleteAnalysisResult, String>>>,

    /// Path to loaded file
    loaded_file: Option<PathBuf>,

    /// Show detailed results
    show_details: bool,
}

impl LottoApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            confidence_level: ConfidenceLevel::High,
            ..Default::default()
        }
    }

    fn load_file(&mut self, path: PathBuf) {
        self.status_message = format!("Loading file: {:?}...", path.file_name().unwrap());

        match DataLoader::load_from_file(&path) {
            Ok(dataset) => {
                self.dataset = Some(dataset.clone());
                self.loaded_file = Some(path.clone());
                self.status_message = format!(
                    "✓ Loaded {} draws from {:?}",
                    dataset.draws.len(),
                    path.file_name().unwrap()
                );
                self.analysis_result = None;
            }
            Err(e) => {
                self.status_message = format!("✗ Error loading file: {}", e);
            }
        }
    }

    fn start_analysis(&mut self) {
        if let Some(dataset) = self.dataset.clone() {
            self.analyzing = true;
            self.status_message = "🔄 Running analysis...".to_string();

            let confidence_level = self.confidence_level;
            let (tx, rx): (Sender<Result<CompleteAnalysisResult, String>>, _) = channel();
            self.result_receiver = Some(rx);

            thread::spawn(move || {
                let analyzer = LottoAnalyzer::new(dataset);
                let result = analyzer.analyze(confidence_level)
                    .map_err(|e| e.to_string());
                let _ = tx.send(result);
            });
        }
    }

    fn check_analysis_result(&mut self) {
        if let Some(rx) = &self.result_receiver {
            if let Ok(result) = rx.try_recv() {
                self.analyzing = false;
                match result {
                    Ok(analysis) => {
                        self.analysis_result = Some(analysis);
                        self.status_message = "✓ Analysis complete!".to_string();
                    }
                    Err(e) => {
                        self.status_message = format!("✗ Analysis error: {}", e);
                    }
                }
                self.result_receiver = None;
            }
        }
    }
}

impl eframe::App for LottoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for analysis completion
        self.check_analysis_result();

        // Request repaint if analyzing
        if self.analyzing {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🎯 LOTTO Analyzer");
            ui.label("Multi-dimensional Mathematical Analysis Engine");
            ui.separator();

            // File loading section
            ui.horizontal(|ui| {
                if ui.button("📁 Load Data File").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Data Files", &["csv", "xlsx", "xls", "dat", "db", "sqlite"])
                        .pick_file()
                    {
                        self.load_file(path);
                    }
                }

                if let Some(path) = &self.loaded_file {
                    ui.label(format!("Current: {:?}", path.file_name().unwrap()));
                }
            });

            ui.separator();

            // Configuration section
            if self.dataset.is_some() {
                ui.heading("⚙️ Configuration");

                ui.horizontal(|ui| {
                    ui.label("Confidence Level:");
                    ui.radio_value(&mut self.confidence_level, ConfidenceLevel::Low, "80% (Low)");
                    ui.radio_value(&mut self.confidence_level, ConfidenceLevel::Medium, "90% (Medium)");
                    ui.radio_value(&mut self.confidence_level, ConfidenceLevel::High, "95% (High)");
                    ui.radio_value(&mut self.confidence_level, ConfidenceLevel::VeryHigh, "99% (Very High)");
                });

                ui.separator();

                // Analysis button
                ui.horizontal(|ui| {
                    let analyze_btn = ui.add_enabled(
                        !self.analyzing,
                        egui::Button::new("🚀 Run Complete Analysis")
                    );

                    if analyze_btn.clicked() {
                        self.start_analysis();
                    }

                    if self.analyzing {
                        ui.spinner();
                        ui.label("Analyzing...");
                    }
                });
            }

            ui.separator();

            // Status message
            if !self.status_message.is_empty() {
                ui.colored_label(
                    if self.status_message.starts_with('✓') {
                        egui::Color32::GREEN
                    } else if self.status_message.starts_with('✗') {
                        egui::Color32::RED
                    } else {
                        egui::Color32::YELLOW
                    },
                    &self.status_message
                );
            }

            ui.separator();

            // Results section
            if let Some(result) = &self.analysis_result {
                ui.heading("📊 Analysis Results");

                ui.checkbox(&mut self.show_details, "Show detailed results");

                ui.separator();

                // Recommended combination
                ui.group(|ui| {
                    ui.heading("🎯 Recommended Combination");
                    ui.horizontal(|ui| {
                        for &num in &result.critical_path.recommended_combination {
                            ui.label(
                                egui::RichText::new(format!("{:02}", num))
                                    .size(24.0)
                                    .strong()
                                    .color(egui::Color32::from_rgb(0, 150, 255))
                            );
                        }
                    });
                    ui.label(format!("Quality Score: {:.4}", result.critical_path.score));
                });

                ui.separator();

                // Combination properties
                egui::CollapsingHeader::new("📈 Combination Properties")
                    .default_open(true)
                    .show(ui, |ui| {
                        let props = &result.critical_path.properties;
                        ui.horizontal(|ui| {
                            ui.label(format!("Sum: {}", props.sum));
                            ui.separator();
                            ui.label(format!("Mean: {:.2}", props.mean));
                            ui.separator();
                            ui.label(format!("Median: {:.1}", props.median));
                            ui.separator();
                            ui.label(format!("Std Dev: {:.2}", props.std_dev));
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("Even: {} | Odd: {}", props.even_count, props.odd_count));
                            ui.separator();
                            ui.label(format!("Primes: {} ({:?})", props.prime_count, props.primes));
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("Low: {} | Med: {} | High: {}",
                                props.low_count, props.medium_count, props.high_count));
                        });
                    });

                if self.show_details {
                    ui.separator();

                    // Detailed analysis results
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        // Frequentist
                        egui::CollapsingHeader::new("1️⃣ Frequentist Analysis (χ²)")
                            .show(ui, |ui| {
                                ui.label(format!("χ² = {:.4}", result.frequentist.chi_squared));
                                ui.label(format!("p-value = {:.4}", result.frequentist.p_value));
                                ui.label(format!("Uniform distribution: {}", result.frequentist.is_uniform));
                            });

                        // Bayesian
                        egui::CollapsingHeader::new("2️⃣ Bayesian Inference (Dirichlet)")
                            .show(ui, |ui| {
                                ui.label("Top 5 numbers by posterior:");
                                for (i, &num) in result.bayesian.top_numbers.iter().take(5).enumerate() {
                                    let prob = result.bayesian.posterior_probabilities
                                        .iter()
                                        .find(|(n, _)| *n == num)
                                        .map(|(_, p)| p)
                                        .unwrap_or(&0.0);
                                    ui.label(format!("  {}. Number {} (p = {:.4})", i + 1, num, prob));
                                }
                            });

                        // Ergodic
                        egui::CollapsingHeader::new("3️⃣ Ergodic Analysis")
                            .show(ui, |ui| {
                                ui.label(format!("Temporal mean: {:.4}", result.ergodic.temporal_mean));
                                ui.label(format!("Ensemble mean: {:.4}", result.ergodic.ensemble_mean));
                                ui.label(format!("Z-score: {:.4}", result.ergodic.runs_test_z_score));
                                ui.label(format!("Is ergodic: {}", result.ergodic.is_ergodic));
                            });

                        // Range Equilibrium
                        egui::CollapsingHeader::new("4️⃣ Range Equilibrium")
                            .show(ui, |ui| {
                                ui.label(format!("Low: {:.1}% | Medium: {:.1}% | High: {:.1}%",
                                    result.range_equilibrium.low_percentage,
                                    result.range_equilibrium.medium_percentage,
                                    result.range_equilibrium.high_percentage));
                                ui.label(format!("Even: {:.1}% | Odd: {:.1}%",
                                    result.range_equilibrium.even_percentage,
                                    result.range_equilibrium.odd_percentage));
                                ui.label(format!("Balanced: {}", result.range_equilibrium.is_balanced));
                            });

                        // Galois Field
                        egui::CollapsingHeader::new("5️⃣ Galois Field GF(29)")
                            .show(ui, |ui| {
                                ui.label(format!("Generator: {}", result.galois_field.generator));
                                ui.label(format!("Quadratic residues: {} elements", result.galois_field.quadratic_residues.len()));
                                ui.label(format!("Combinatorial dimension: {:.2} bits", result.galois_field.combinatorial_dimension));
                            });

                        // Multifractal
                        egui::CollapsingHeader::new("6️⃣ Multifractal Analysis")
                            .show(ui, |ui| {
                                ui.label(format!("Hurst exponent: {:.4}", result.multifractal.hurst_exponent));
                                ui.label(format!("Fractal dimension: {:.4}", result.multifractal.fractal_dimension));
                                ui.label(format!("Interpretation: {}", result.multifractal.interpretation));
                            });

                        // Chaos Theory
                        egui::CollapsingHeader::new("7️⃣ Chaos Theory")
                            .show(ui, |ui| {
                                ui.label(format!("Lyapunov exponent: {:.4}", result.chaos_theory.lyapunov_exponent));
                                ui.label(format!("Correlation dimension: {:.4}", result.chaos_theory.correlation_dimension));
                                ui.label(format!("Is chaotic: {}", result.chaos_theory.is_chaotic));
                            });

                        // Normalization
                        egui::CollapsingHeader::new("8️⃣ Normalization")
                            .show(ui, |ui| {
                                ui.label(format!("Outliers: {:?}", result.normalization.outliers));
                                ui.label(format!("Shapiro-Wilk W: {:.4} (p = {:.4})",
                                    result.normalization.shapiro_wilk_w,
                                    result.normalization.shapiro_wilk_p));
                                ui.label(format!("Box-Cox λ: {:.4}", result.normalization.box_cox_lambda));
                            });
                    });
                }

                ui.separator();

                // Export button
                if ui.button("💾 Export Results (JSON)").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("JSON", &["json"])
                        .set_file_name("lotto_analysis.json")
                        .save_file()
                    {
                        match std::fs::write(&path, serde_json::to_string_pretty(result).unwrap()) {
                            Ok(_) => self.status_message = format!("✓ Exported to {:?}", path.file_name().unwrap()),
                            Err(e) => self.status_message = format!("✗ Export error: {}", e),
                        }
                    }
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("LOTTO Analyzer v1.0.0");
                ui.separator();
                ui.label("© 2025 Francisco Molina");
            });
        });
    }
}

pub fn run_app() -> Result<(), eframe::Error> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("LOTTO Analyzer - Multi-dimensional Analysis Engine"),
        ..Default::default()
    };

    eframe::run_native(
        "LOTTO Analyzer",
        options,
        Box::new(|cc| Ok(Box::new(LottoApp::new(cc)))),
    )
}
