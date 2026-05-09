package promptcompat

import (
	"encoding/json"
	"fmt"
	"strings"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

const inlineStructuredOutputSchemaResource = "inline-structured-output-schema.json"

type StructuredOutputSpec struct {
	Mode   string
	Name   string
	Schema any
	Strict bool
}

func ExtractStructuredOutputSpecFromChat(req map[string]any) *StructuredOutputSpec {
	return extractStructuredOutputSpec(req["response_format"])
}

func ExtractStructuredOutputSpecFromResponses(req map[string]any) *StructuredOutputSpec {
	if textObj, ok := req["text"].(map[string]any); ok {
		if format, ok := textObj["format"]; ok {
			if spec := extractStructuredOutputSpec(format); spec != nil {
				return spec
			}
		}
	}
	return extractStructuredOutputSpec(req["text_format"])
}

func extractStructuredOutputSpec(raw any) *StructuredOutputSpec {
	m, ok := raw.(map[string]any)
	if !ok || len(m) == 0 {
		return nil
	}
	typ := strings.ToLower(strings.TrimSpace(asString(m["type"])))
	switch typ {
	case "", "text":
		return nil
	case "json_object":
		return &StructuredOutputSpec{Mode: "json_object"}
	case "json_schema":
		schemaObj, _ := m["json_schema"].(map[string]any)
		if len(schemaObj) == 0 {
			schemaObj = m
		}
		return &StructuredOutputSpec{
			Mode:   "json_schema",
			Name:   strings.TrimSpace(asString(schemaObj["name"])),
			Schema: schemaObj["schema"],
			Strict: toBool(schemaObj["strict"]),
		}
	default:
		return nil
	}
}

func ValidateStructuredOutputSpec(spec *StructuredOutputSpec) error {
	if spec == nil {
		return nil
	}
	switch spec.Mode {
	case "json_object":
		return nil
	case "json_schema":
		if spec.Schema == nil {
			return fmt.Errorf("response_format json_schema requires a schema")
		}
		_, err := compileStructuredOutputSchema(spec.Schema)
		if err != nil {
			return fmt.Errorf("invalid response_format json_schema: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("unsupported structured output mode %q", spec.Mode)
	}
}

func ApplyStructuredOutputPrompt(messagesRaw []any, spec *StructuredOutputSpec) []any {
	if spec == nil {
		return messagesRaw
	}
	systemMsg := map[string]any{"role": "system", "content": buildStructuredOutputInstruction(spec)}
	out := make([]any, 0, len(messagesRaw)+1)
	out = append(out, systemMsg)
	out = append(out, messagesRaw...)
	return out
}

func buildStructuredOutputInstruction(spec *StructuredOutputSpec) string {
	parts := []string{
		"You must respond with exactly one valid JSON value and nothing else.",
		"Do not include markdown fences, comments, prose, headings, or any surrounding text.",
		"Do not explain your answer.",
	}
	switch spec.Mode {
	case "json_object":
		parts = append(parts, "The top-level JSON value must be an object ({}).")
	case "json_schema":
		parts = append(parts,
			"The JSON must satisfy every constraint in the provided JSON Schema.",
			"Include every required field.",
			"Do not add fields or values that would violate the schema.",
		)
		if spec.Strict {
			parts = append(parts, "Treat the schema as strict: every constraint is mandatory and must be satisfied exactly.")
		}
		if spec.Schema != nil {
			if schemaBytes, err := json.Marshal(spec.Schema); err == nil {
				parts = append(parts, "JSON_SCHEMA_BEGIN "+string(schemaBytes)+" JSON_SCHEMA_END")
			}
		}
	}
	return strings.Join(parts, " ")
}

// ValidateStructuredOutput strictly validates text against spec.
// No coercion, no fallback — returns error if the text is not valid JSON matching the spec.
func ValidateStructuredOutput(text string, spec *StructuredOutputSpec) (string, any, error) {
	if spec == nil {
		return "", nil, fmt.Errorf("structured output is not enabled")
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return "", nil, fmt.Errorf("empty structured output")
	}
	parsed, ok := parseStructuredJSONFromText(trimmed)
	if !ok {
		return "", nil, fmt.Errorf("reply did not contain valid JSON")
	}
	switch spec.Mode {
	case "json_object":
		obj, ok := parsed.(map[string]any)
		if !ok || obj == nil {
			return "", nil, fmt.Errorf("reply must be a JSON object, got %T", parsed)
		}
		b, err := json.Marshal(obj)
		if err != nil {
			return "", nil, fmt.Errorf("marshal structured output: %w", err)
		}
		return string(b), obj, nil
	case "json_schema":
		compiled, err := compileStructuredOutputSchema(spec.Schema)
		if err != nil {
			return "", nil, fmt.Errorf("invalid response_format json_schema: %w", err)
		}
		if err := compiled.Validate(parsed); err != nil {
			return "", nil, fmt.Errorf("reply does not satisfy json_schema: %w", err)
		}
		b, err := json.Marshal(parsed)
		if err != nil {
			return "", nil, fmt.Errorf("marshal structured output: %w", err)
		}
		return string(b), parsed, nil
	default:
		return "", nil, fmt.Errorf("unsupported structured output mode %q", spec.Mode)
	}
}

// CanonicalizeStructuredOutput is a convenience wrapper around ValidateStructuredOutput.
func CanonicalizeStructuredOutput(text string, spec *StructuredOutputSpec) (string, any, bool) {
	if spec == nil {
		return "", nil, false
	}
	canonical, parsed, err := ValidateStructuredOutput(text, spec)
	if err != nil {
		return "", nil, false
	}
	return canonical, parsed, true
}

func parseStructuredJSONFromText(raw string) (any, bool) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, false
	}
	if parsed, ok := decodeJSON(trimmed); ok {
		return parsed, true
	}
	fenceStart := strings.Index(trimmed, "```")
	if fenceStart >= 0 {
		fenced := trimmed[fenceStart+3:]
		fenced = strings.TrimSpace(strings.TrimPrefix(fenced, "json"))
		if end := strings.Index(fenced, "```"); end >= 0 {
			if parsed, ok := decodeJSON(strings.TrimSpace(fenced[:end])); ok {
				return parsed, true
			}
		}
	}
	if candidate, ok := firstBalancedJSON(trimmed); ok {
		return decodeJSON(candidate)
	}
	return nil, false
}

func decodeJSON(raw string) (any, bool) {
	var parsed any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, false
	}
	return parsed, true
}

func firstBalancedJSON(raw string) (string, bool) {
	start := -1
	for i, r := range raw {
		if r == '{' || r == '[' {
			start = i
			break
		}
	}
	if start < 0 {
		return "", false
	}
	opener := raw[start]
	closer := byte('}')
	if opener == '[' {
		closer = ']'
	}
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(raw); i++ {
		ch := raw[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			continue
		}
		if ch == opener {
			depth++
		}
		if ch == closer {
			depth--
			if depth == 0 {
				return raw[start : i+1], true
			}
		}
	}
	return "", false
}

func compileStructuredOutputSchema(schema any) (*jsonschema.Schema, error) {
	// jsonschema/v6 AddResource expects a JSON-compatible Go value (map/slice/etc.), not a reader.
	// Round-trip through JSON to normalise any non-standard types before passing it in.
	var normalized any
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}
	if err := json.Unmarshal(b, &normalized); err != nil {
		return nil, fmt.Errorf("unmarshal schema: %w", err)
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource(inlineStructuredOutputSchemaResource, normalized); err != nil {
		return nil, fmt.Errorf("load schema: %w", err)
	}
	compiled, err := compiler.Compile(inlineStructuredOutputSchemaResource)
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}
	return compiled, nil
}

func toBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return strings.EqualFold(strings.TrimSpace(x), "true")
	default:
		return false
	}
}
