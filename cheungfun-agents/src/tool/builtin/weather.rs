//! Weather Tool - Get weather information
//!
//! This tool provides weather information using a weather API service.

use crate::{
    error::{AgentError, Result},
    tool::{create_simple_schema, Tool, ToolContext, ToolResult},
    types::ToolSchema,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Weather tool for getting weather information
#[derive(Debug, Clone)]
pub struct WeatherTool {
    name: String,
    api_key: Option<String>,
}

impl WeatherTool {
    /// Create a new weather tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "weather".to_string(),
            api_key: None,
        }
    }

    /// Create weather tool with API key
    #[must_use]
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            name: "weather".to_string(),
            api_key: Some(api_key.into()),
        }
    }
}

impl Default for WeatherTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WeatherResponse {
    location: String,
    temperature: f64,
    condition: String,
    humidity: f64,
    wind_speed: f64,
    description: String,
}

#[async_trait]
impl Tool for WeatherTool {
    fn schema(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "location".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Location to get weather for (city name, coordinates, etc.)"
            }),
        );
        properties.insert(
            "units".to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Temperature units: 'celsius', 'fahrenheit'",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }),
        );

        ToolSchema {
            name: self.name.clone(),
            description: "Get current weather information for a specified location.".to_string(),
            input_schema: create_simple_schema(properties, vec!["location".to_string()]),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location name"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Current temperature"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Weather condition (sunny, cloudy, rainy, etc.)"
                    },
                    "humidity": {
                        "type": "number",
                        "description": "Humidity percentage"
                    },
                    "wind_speed": {
                        "type": "number",
                        "description": "Wind speed"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed weather description"
                    }
                }
            })),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct WeatherArgs {
            location: String,
            #[serde(default = "default_units")]
            units: String,
        }

        fn default_units() -> String {
            "celsius".to_string()
        }

        let args: WeatherArgs = serde_json::from_value(arguments)
            .map_err(|e| AgentError::tool(&self.name, format!("Invalid arguments: {e}")))?;

        // In a real implementation, you would call an actual weather API
        // For demo purposes, we'll return mock data
        let weather_data = self.get_mock_weather_data(&args.location, &args.units)?;

        let content = format!(
            "Weather in {}: {}°{}, {}. Humidity: {}%, Wind: {} km/h. {}",
            weather_data.location,
            weather_data.temperature,
            if args.units == "celsius" { "C" } else { "F" },
            weather_data.condition,
            weather_data.humidity,
            weather_data.wind_speed,
            weather_data.description
        );

        let result = ToolResult::success(content)
            .with_metadata(
                "location".to_string(),
                serde_json::json!(weather_data.location),
            )
            .with_metadata(
                "temperature".to_string(),
                serde_json::json!(weather_data.temperature),
            )
            .with_metadata(
                "condition".to_string(),
                serde_json::json!(weather_data.condition),
            )
            .with_metadata(
                "humidity".to_string(),
                serde_json::json!(weather_data.humidity),
            )
            .with_metadata(
                "wind_speed".to_string(),
                serde_json::json!(weather_data.wind_speed),
            )
            .with_metadata(
                "description".to_string(),
                serde_json::json!(weather_data.description),
            )
            .with_metadata("units".to_string(), serde_json::json!(args.units));

        Ok(result)
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "weather".to_string(),
            "information".to_string(),
            "external_api".to_string(),
        ]
    }
}

impl WeatherTool {
    /// Get mock weather data (replace with real API call)
    fn get_mock_weather_data(&self, location: &str, units: &str) -> Result<WeatherResponse> {
        // This is mock data. In a real implementation, you would:
        // 1. Make an HTTP request to a weather API (e.g., OpenWeatherMap)
        // 2. Parse the response
        // 3. Convert units as needed

        let base_temp = match location.to_lowercase().as_str() {
            "london" | "uk" | "england" => 12.0,
            "tokyo" | "japan" => 18.0,
            "new york" | "nyc" => 15.0,
            "paris" | "france" => 14.0,
            "berlin" | "germany" => 11.0,
            "sydney" | "australia" => 22.0,
            "beijing" | "china" => 16.0,
            "moscow" | "russia" => -2.0,
            "cairo" | "egypt" => 28.0,
            "mumbai" | "india" => 30.0,
            _ => 20.0, // Default temperature
        };

        let temperature = if units == "fahrenheit" {
            base_temp * 9.0 / 5.0 + 32.0
        } else {
            base_temp
        };

        let conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"];
        let condition = conditions[location.len() % conditions.len()];

        Ok(WeatherResponse {
            location: location.to_string(),
            temperature,
            condition: condition.to_string(),
            humidity: 45.0 + (location.len() as f64 % 40.0),
            wind_speed: 5.0 + (location.len() as f64 % 20.0),
            description: format!("{condition} with comfortable conditions"),
        })
    }

    /// Set API key for real weather service
    pub fn set_api_key(&mut self, api_key: impl Into<String>) {
        self.api_key = Some(api_key.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_weather_tool() {
        let tool = WeatherTool::new();
        let context = ToolContext::new();

        // Test basic weather request
        let args = serde_json::json!({
            "location": "London",
            "units": "celsius"
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("London"));
        assert!(result.content.contains("°C"));

        // Check metadata
        assert!(result.metadata.contains_key("location"));
        assert!(result.metadata.contains_key("temperature"));
        assert!(result.metadata.contains_key("condition"));
    }

    #[tokio::test]
    async fn test_weather_fahrenheit() {
        let tool = WeatherTool::new();
        let context = ToolContext::new();

        // Test fahrenheit units
        let args = serde_json::json!({
            "location": "New York",
            "units": "fahrenheit"
        });

        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("°F"));
    }
}
