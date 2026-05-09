package promptcompat

import (
	"strings"
	"testing"
)

type structuredOutputTestConfig struct{}

func (structuredOutputTestConfig) ModelAliases() map[string]string { return nil }

func TestExtractStructuredOutputSpecFromResponses(t *testing.T) {
	req := map[string]any{
		"text": map[string]any{
			"format": map[string]any{
				"type": "json_schema",
				"json_schema": map[string]any{
					"name":   "Quiz",
					"strict": true,
					"schema": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"title": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
	}
	spec := ExtractStructuredOutputSpecFromResponses(req)
	if spec == nil || spec.Mode != "json_schema" || spec.Name != "Quiz" || !spec.Strict {
		t.Fatalf("unexpected spec: %#v", spec)
	}
}

func TestValidateStructuredOutputWithSchema(t *testing.T) {
	spec := &StructuredOutputSpec{
		Mode: "json_schema",
		Schema: map[string]any{
			"type":                 "object",
			"required":             []any{"title", "score"},
			"additionalProperties": false,
			"properties": map[string]any{
				"title": map[string]any{"type": "string"},
				"score": map[string]any{"type": "integer"},
			},
		},
	}
	out, parsed, err := ValidateStructuredOutput("```json\n{\"title\":\"quiz\",\"score\":9}\n```", spec)
	if err != nil {
		t.Fatalf("expected validation success, got %v", err)
	}
	if out == "" {
		t.Fatal("expected non-empty canonical output")
	}
	obj, _ := parsed.(map[string]any)
	if obj["title"] != "quiz" {
		t.Fatalf("expected parsed title=quiz, got %#v", obj["title"])
	}
}

func TestValidateStructuredOutputRejectsSchemaMismatch(t *testing.T) {
	spec := &StructuredOutputSpec{
		Mode: "json_schema",
		Schema: map[string]any{
			"type":                 "object",
			"required":             []any{"title", "score"},
			"additionalProperties": false,
			"properties": map[string]any{
				"title": map[string]any{"type": "string"},
				"score": map[string]any{"type": "integer"},
			},
		},
	}
	// score is a string, not integer — must fail
	_, _, err := ValidateStructuredOutput(`{"title":"quiz","score":"9"}`, spec)
	if err == nil {
		t.Fatal("expected validation failure for type mismatch")
	}
	if !strings.Contains(err.Error(), "json_schema") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateStructuredOutputRejectsNonObjectJSONMode(t *testing.T) {
	spec := &StructuredOutputSpec{Mode: "json_object"}
	_, _, err := ValidateStructuredOutput(`[1,2,3]`, spec)
	if err == nil {
		t.Fatal("expected json_object validation failure for array input")
	}
}

func TestValidateStructuredOutputSpecRejectsMissingSchema(t *testing.T) {
	err := ValidateStructuredOutputSpec(&StructuredOutputSpec{Mode: "json_schema"})
	if err == nil {
		t.Fatal("expected error for json_schema mode with nil schema")
	}
}

func TestValidateStructuredOutputNoCoercion(t *testing.T) {
	spec := &StructuredOutputSpec{
		Mode: "json_schema",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"score": map[string]any{"type": "integer"},
			},
			"required": []any{"score"},
		},
	}
	// score as string should fail — no coercion
	_, _, err := ValidateStructuredOutput(`{"score":"9"}`, spec)
	if err == nil {
		t.Fatal("expected failure: integer field received as string must not be coerced")
	}
}

func TestValidateStructuredOutputNoFallback(t *testing.T) {
	spec := &StructuredOutputSpec{Mode: "json_object"}
	// plain prose — must fail, not return {"text": ...}
	_, _, err := ValidateStructuredOutput("Here is the answer: 42", spec)
	if err == nil {
		t.Fatal("expected failure: prose input must not produce fallback object")
	}
}

func TestValidateStructuredOutputRejectsMissingRequired(t *testing.T) {
	spec := &StructuredOutputSpec{
		Mode: "json_schema",
		Schema: map[string]any{
			"type":     "object",
			"required": []any{"title", "score"},
			"properties": map[string]any{
				"title": map[string]any{"type": "string"},
				"score": map[string]any{"type": "integer"},
			},
		},
	}
	// missing required field "score"
	_, _, err := ValidateStructuredOutput(`{"title":"quiz"}`, spec)
	if err == nil {
		t.Fatal("expected failure: missing required field must be rejected")
	}
}

func TestNormalizeOpenAIChatRequestRejectsInvalidStructuredOutputSchema(t *testing.T) {
	_, err := NormalizeOpenAIChatRequest(structuredOutputTestConfig{}, map[string]any{
		"model": "deepseek-v4-flash",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "BadSchema",
				"strict": true,
				"schema": map[string]any{
					"type": 123,
				},
			},
		},
	}, "")
	if err == nil {
		t.Fatal("expected invalid schema error")
	}
	if !strings.Contains(err.Error(), "invalid response_format json_schema") {
		t.Fatalf("unexpected error: %v", err)
	}
}
