use clap::{Parser, Subcommand};
use colored::*;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use linal::server::start_server;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::sync::{Arc, Mutex};
use toon_format::encode_default;

#[derive(Parser)]
#[command(name = "LINAL")]
#[command(version = "0.1")]
#[command(about = "LINAL: Linear Algebra Analytical Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start REPL (default)
    Repl {
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Run a script file
    Run {
        /// Path to the script file (.lnl)
        file: String,
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Start HTTP server
    Server {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Start HTTP server (shorthand for server)
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Initialize a new LINAL project structure
    Init,
    /// Load a Parquet file directly into a dataset
    Load {
        /// Path to the parquet file
        file: String,
        /// Target dataset name
        dataset: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut db = TensorDb::new();

    match cli.command {
        Some(Commands::Run { file, format }) => {
            let content = fs::read_to_string(&file)?;
            let use_toon = format == "toon";

            let mut current_cmd = String::new();
            let mut start_line = 0;
            let mut paren_balance = 0;

            for (idx, raw_line) in content.lines().enumerate() {
                let line = raw_line.trim();

                if current_cmd.is_empty() {
                    if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                        continue;
                    }
                    start_line = idx + 1;
                }

                if !current_cmd.is_empty() {
                    current_cmd.push(' ');
                }
                current_cmd.push_str(line);

                for c in line.chars() {
                    if c == '(' {
                        paren_balance += 1;
                    } else if c == ')' {
                        paren_balance -= 1;
                    }
                }

                if paren_balance == 0 {
                    match execute_line(&mut db, &current_cmd, start_line) {
                        Ok(output) => {
                            if !matches!(output, DslOutput::None) {
                                if use_toon {
                                    let toon = encode_default(&output)
                                        .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                    println!("{}", toon);
                                } else {
                                    println!("{}", output);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error on line {}: {}", start_line, e);
                            std::process::exit(1);
                        }
                    }
                    current_cmd.clear();
                }
            }

            if !current_cmd.is_empty() {
                eprintln!(
                    "Error: Script ended with unbalanced parentheses starting at line {}",
                    start_line
                );
                std::process::exit(1);
            }
        }
        Some(Commands::Server { port }) | Some(Commands::Serve { port }) => {
            // Need Arc<Mutex<TensorDb>>
            let db_arc = Arc::new(Mutex::new(db));
            start_server(db_arc, port).await;
        }
        Some(Commands::Init) => {
            handle_init()?;
        }
        Some(Commands::Load { file, dataset }) => {
            handle_load(&mut db, &file, &dataset)?;
        }
        Some(Commands::Repl { format }) => {
            run_repl(db, format == "toon")?;
        }
        None => {
            run_repl(db, false)?;
        }
    }

    Ok(())
}

fn handle_init() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = "./data";
    if !std::path::Path::new(data_dir).exists() {
        fs::create_dir_all(data_dir)?;
        println!("Created directory: {}", data_dir.green());
    }

    let config_path = "linal.toml";
    if !std::path::Path::new(config_path).exists() {
        let default_config = r#"[storage]
data_dir = "./data"
default_db = "default"
"#;
        fs::write(config_path, default_config)?;
        println!("Created default configuration: {}", config_path.green());
    } else {
        println!(
            "Configuration file already exists: {}",
            config_path.yellow()
        );
    }

    println!(
        "{}",
        "Initialization complete. Welcome to LINAL!".bold().blue()
    );
    Ok(())
}

fn handle_load(
    db: &mut TensorDb,
    file: &str,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let command = format!("LOAD DATASET {} FROM \"{}\"", dataset, file);
    match execute_line(db, &command, 1) {
        Ok(output) => {
            println!("{}", output.to_string().green());
            Ok(())
        }
        Err(e) => {
            eprintln!("{}: {}", "Error loading dataset".red(), e);
            Err(e.into())
        }
    }
}

fn run_repl(mut db: TensorDb, use_toon: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rl = DefaultEditor::new()?;
    let history_path = ".linal_history";

    if rl.load_history(history_path).is_err() {
        // No history yet
    }

    println!("{}", "LINAL REPL v0.1".bold().blue());
    if use_toon {
        println!("Output format: {}", "TOON (machine-readable)".yellow());
    } else {
        println!("Output format: {}", "Display (human-readable)".yellow());
    }
    println!("Type 'EXIT' or use Ctrl-D to quit.");

    let mut current_cmd = String::new();
    let mut paren_balance = 0;

    loop {
        let prompt = if paren_balance == 0 { ">_>  " } else { " ..  " };
        let readline = rl.readline(prompt);

        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.eq_ignore_ascii_case("EXIT") {
                    break;
                }

                rl.add_history_entry(trimmed)?;

                if !current_cmd.is_empty() {
                    current_cmd.push(' ');
                }
                current_cmd.push_str(trimmed);

                for c in trimmed.chars() {
                    if c == '(' {
                        paren_balance += 1;
                    } else if c == ')' {
                        paren_balance -= 1;
                    }
                }

                if paren_balance == 0 {
                    match execute_line(&mut db, &current_cmd, 1) {
                        Ok(output) => {
                            if !matches!(output, DslOutput::None) {
                                if use_toon {
                                    let toon = encode_default(&output)
                                        .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                    println!("{}", toon);
                                } else {
                                    println!("{}", output);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("{}: {}", "Error".red(), e);
                        }
                    }
                    current_cmd.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("Interrupted");
                current_cmd.clear();
                paren_balance = 0;
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(history_path);
    Ok(())
}
