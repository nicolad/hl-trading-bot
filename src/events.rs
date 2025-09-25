use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use chrono::{DateTime, Utc};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EventType {
    OrderFilled,
    OrderCancelled,
    OrderPlaced,
    PositionOpened,
    PositionClosed,
    PositionUpdated,
    PriceUpdate,
    StrategyStart,
    StrategyStop,
    StrategyUpdate,
    Error,
    System,
    EmergencyStop,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
    pub source: Option<String>,
}

type Listener = Arc<dyn Fn(&Event) + Send + Sync>;

#[derive(Clone, Default)]
pub struct EventBus {
    listeners: Arc<Mutex<HashMap<EventType, Vec<Listener>>>>,
}

impl EventBus {
    pub fn subscribe<F>(&self, event_type: EventType, callback: F)
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        let mut guard = self.listeners.lock().expect("event bus poisoned");
        guard
            .entry(event_type)
            .or_default()
            .push(Arc::new(callback));
    }

    pub fn unsubscribe_all(&self, event_type: EventType) {
        if let Ok(mut guard) = self.listeners.lock() {
            guard.remove(&event_type);
        }
    }

    pub fn emit(&self, event: Event) {
        if let Ok(guard) = self.listeners.lock()
            && let Some(listeners) = guard.get(&event.event_type)
        {
            for listener in listeners {
                listener(&event);
            }
        }
    }
}
