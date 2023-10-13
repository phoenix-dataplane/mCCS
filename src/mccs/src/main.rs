use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use nix::sys::signal;

use anyhow::Result;
use structopt::StructOpt;

use mccs::config::Config;
use mccs::control::Control;

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "mCCS Service")]
struct Opts {
    /// Phoenix config path
    #[structopt(short, long, default_value = "mccs.toml")]
    config: PathBuf,
}

static TERMINATE: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(sig: i32) {
    assert_eq!(sig, signal::SIGINT as i32);
    TERMINATE.store(true, Ordering::Relaxed);
}

fn main() -> Result<()> {
    // load config
    let opts = Opts::from_args();
    let config = Config::from_path(opts.config)?;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // process Ctrl-C event
    let sig_action = signal::SigAction::new(
        signal::SigHandler::Handler(handle_sigint),
        signal::SaFlags::empty(),
        signal::SigSet::empty(),
    );
    unsafe { signal::sigaction(signal::SIGINT, &sig_action) }
        .expect("failed to register sighandler");

    // the Control now takes over
    let mut control = Control::new(config);
    log::info!("Started mCCS");
    control.mainloop(&TERMINATE)
}
