package promptcompat

import (
	"encoding/json"
	"strings"
)

// ResponseFormatType is the type of response format requested by the client.
type ResponseFormatType string

const (
	// ResponseFormatText is the default plain-text response (no special output constraint).
	ResponseFormatText ResponseFormatType = "text"
	// ResponseFormatJSONObject requests free-form valid JSON output (legacy mode).
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	// ResponseFormatJSONSchema requests JSON output strictly conforming to a provided schema.
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// JSONSchemaFormat holds the metadata from a json_schema response_format request.
type JSONSchemaFormat struct {
	Name        string
	Description string
	Schema      any
	Strict      bool
}

// ResponseFormat represents a parsed response_format (Chat API) or text.format (Responses API).
type ResponseFormat struct {
	Type       ResponseFormatType
	JSONSchema *JSONSchemaFormat
}

// IsJSON returns true when the format constrains output to JSON
// (either json_object or json_schema).
func (rf *ResponseFormat) IsJSON() bool {
	if rf == nil {
		return false
	}
	return rf.Type == ResponseFormatJSONObject || rf.Type == ResponseFormatJSONSchema
}

// ParseResponseFormat parses a raw response_format value from an OpenAI Chat request body.
// The raw value is the map at req["response_format"].
func ParseResponseFormat(raw any) *ResponseFormat {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil
	}
	typ := strings.ToLower(strings.TrimSpace(asString(m["type"])))
	switch typ {
	case "text":
		return &ResponseFormat{Type: ResponseFormatText}
	case "json_object":
		return &ResponseFormat{Type: ResponseFormatJSONObject}
	case "json_schema":
		jsMeta, _ := m["json_schema"].(map[string]any)
		if jsMeta == nil {
			// json_schema without the inner object: treat as unstructured json_schema
			return &ResponseFormat{Type: ResponseFormatJSONSchema}
		}
		strict, _ := jsMeta["strict"].(bool)
		return &ResponseFormat{
			Type: ResponseFormatJSONSchema,
			JSONSchema: &JSONSchemaFormat{
				Name:        strings.TrimSpace(asString(jsMeta["name"])),
				Description: strings.TrimSpace(asString(jsMeta["description"])),
				Schema:      jsMeta["schema"],
				Strict:      strict,
			},
		}
	}
	return nil
}

// ParseResponsesTextFormat parses the format from the Responses API text.format field.
// The raw value is the map at req["text"]["format"].
func ParseResponsesTextFormat(req map[string]any) *ResponseFormat {
	textRaw, ok := req["text"].(map[string]any)
	if !ok {
		return nil
	}
	return ParseResponseFormat(textRaw["format"])
}

// BuildResponseFormatInstruction returns a system-prompt instruction string that
// tells the model to output JSON conforming to the response format. Returns an
// empty string for nil or text format.
func BuildResponseFormatInstruction(rf *ResponseFormat) string {
	if rf == nil || !rf.IsJSON() {
		return ""
	}
	if rf.Type == ResponseFormatJSONObject {
		return "Output format: Respond ONLY with a valid JSON object. " +
			"Output nothing else — no explanation, no markdown fences, no text before or after the JSON."
	}
	// json_schema
	if rf.JSONSchema == nil || rf.JSONSchema.Schema == nil {
		return "Output format: Respond ONLY with a valid JSON object. " +
			"Output nothing else — no explanation, no markdown fences, no text before or after the JSON."
	}
	schemaBytes, err := json.Marshal(rf.JSONSchema.Schema)
	if err != nil {
		return "Output format: Respond ONLY with a valid JSON object. " +
			"Output nothing else — no explanation, no markdown fences, no text before or after the JSON."
	}
	var sb strings.Builder
	sb.WriteString("Output format: Respond ONLY with a valid JSON object that strictly conforms to this JSON Schema.")
	if rf.JSONSchema.Description != "" {
		sb.WriteString("\nDescription: ")
		sb.WriteString(rf.JSONSchema.Description)
	}
	sb.WriteString("\nSchema: ")
	sb.WriteString(string(schemaBytes))
	sb.WriteString("\nOutput nothing else — no explanation, no markdown fences, no text before or after the JSON object.")
	return sb.String()
}

// InjectResponseFormatIntoMessages adds the response format instruction into the
// messages slice by appending to the first system message, or by prepending a new
// system message if none exists. Returns the (possibly modified) messages slice.
func InjectResponseFormatIntoMessages(messages []any, rf *ResponseFormat) []any {
	instruction := BuildResponseFormatInstruction(rf)
	if instruction == "" {
		return messages
	}
	for i, msg := range messages {
		m, ok := msg.(map[string]any)
		if !ok || m["role"] != "system" {
			continue
		}
		old, _ := m["content"].(string)
		newMsg := make(map[string]any, len(m))
		for k, v := range m {
			newMsg[k] = v
		}
		newMsg["content"] = strings.TrimSpace(old+"\n\n"+instruction)
		result := make([]any, len(messages))
		copy(result, messages)
		result[i] = newMsg
		return result
	}
	// No system message found — prepend a new one.
	out := make([]any, 0, len(messages)+1)
	out = append(out, map[string]any{"role": "system", "content": instruction})
	out = append(out, messages...)
	return out
}

// ExtractJSONOutput cleans up model text output that is supposed to contain JSON.
// It strips common wrapping like ```json ... ``` or ``` ... ``` markdown fences,
// returning only the inner JSON content. If no fences are found the input is
// returned trimmed.
func ExtractJSONOutput(text string) string {
	s := strings.TrimSpace(text)
	for _, fence := range []string{"```json\n", "```JSON\n", "```json", "```JSON", "```\n", "```"} {
		if strings.HasPrefix(s, fence) {
			rest := s[len(fence):]
			// Find the last closing fence
			if idx := strings.LastIndex(rest, "```"); idx >= 0 {
				return strings.TrimSpace(rest[:idx])
			}
			// No closing fence found — return the rest trimmed
			return strings.TrimSpace(rest)
		}
	}
	return s
}
