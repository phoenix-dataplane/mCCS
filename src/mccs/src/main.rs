use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use nix::sys::signal;

use anyhow::Result;
use structopt::StructOpt;

use mccs::config::Config;
use mccs::control::Control;

use chrono::Timelike;
use env_logger::fmt::Color;
use std::io::Write;

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "mCCS Service")]
struct Opts {
    /// Phoenix config path
    #[structopt(short, long, default_value = "mccs.toml")]
    config: PathBuf,
    #[structopt(short, long)]
    host: usize,
}

static TERMINATE: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(sig: i32) {
    assert_eq!(sig, signal::SIGINT as i32);
    TERMINATE.store(true, Ordering::Relaxed);
}

fn main() -> Result<()> {
    better_panic::install();
    // load config
    let opts = Opts::from_args();
    let config = Config::from_path(opts.config)?;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            let time = chrono::Local::now();
            let style = buf
                .style()
                .set_color(Color::Black)
                .set_intense(true)
                .clone();
            let time = format!(
                "{:02}:{:02}:{:02}.{:03}",
                time.hour() % 24,
                time.minute(),
                time.second(),
                time.timestamp_subsec_millis()
            );
            writeln!(
                buf,
                "{}{} {} {}{} {}",
                style.value("["),
                time,
                buf.default_styled_level(record.level()),
                record.module_path().unwrap_or(""),
                style.value("]"),
                record.args()
            )
        })
        .init();

    // process Ctrl-C event
    let sig_action = signal::SigAction::new(
        signal::SigHandler::Handler(handle_sigint),
        signal::SaFlags::empty(),
        signal::SigSet::empty(),
    );
    unsafe { signal::sigaction(signal::SIGINT, &sig_action) }
        .expect("failed to register sighandler");

    // the Control now takes over
    let mut control = Control::new(config, opts.host);
    log::info!("Started mCCS");

    control.mainloop(&TERMINATE)
}
