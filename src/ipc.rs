//! Channel types used to bridge the tokio MCP thread and the render thread.

use serde_json::Value;
use tokio::sync::oneshot;

/// A single command sent from the MCP thread to the main render thread.
pub struct AppCommand {
    pub id: u64,
    pub method: String,
    pub params: Value,
    pub reply: oneshot::Sender<Result<Value, String>>,
}

pub type CommandSender = std::sync::mpsc::Sender<AppCommand>;
pub type CommandReceiver = std::sync::mpsc::Receiver<AppCommand>;

pub fn channel() -> (CommandSender, CommandReceiver) {
    std::sync::mpsc::channel()
}
