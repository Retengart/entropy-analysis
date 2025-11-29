//! Services module for modular architecture
//!
//! This module contains the core business logic and services
//! that have been extracted from main.rs for better organization,
//! testability, and maintainability.

pub mod errors;
pub mod text_analyzer;

// Re-export main types for convenience
// Re-exports will be added when modules are used
