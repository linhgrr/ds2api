package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatCompletionsStructuredOutputFailureReturns422WithoutChoices(t *testing.T) {
	h := &openAITestSurface{
		Store: mockOpenAIConfig{},
		Auth:  streamStatusAuthStub{},
		DS: &streamStatusDSSeqStub{resps: []*http.Response{
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":101,"p":"response/content","v":"{\"score\":\"a\"}"}`, `data: [DONE]`),
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":102,"p":"response/content","v":"{\"score\":\"b\"}"}`, `data: [DONE]`),
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":103,"p":"response/content","v":"{\"score\":\"c\"}"}`, `data: [DONE]`),
		}},
	}

	reqBody := `{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Return score=9."}],"stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"ScoreCard","strict":true,"schema":{"type":"object","additionalProperties":false,"required":["score"],"properties":{"score":{"type":"integer"}}}}}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	newOpenAITestRouter(h).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("expected 422, got %d body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if _, ok := body["choices"]; ok {
		t.Fatalf("expected fail-closed error body without choices, got %s", rec.Body.String())
	}
	if _, ok := body["usage"]; ok {
		t.Fatalf("expected fail-closed error body without usage, got %s", rec.Body.String())
	}
	errObj, _ := body["error"].(map[string]any)
	if errObj["code"] != "structured_output_validation_failed" {
		t.Fatalf("unexpected error code: %#v", errObj)
	}
	msg, _ := errObj["message"].(string)
	if !strings.Contains(msg, "Structured output validation failed") {
		t.Fatalf("unexpected error message: %#v", errObj)
	}
}

func TestResponsesStructuredOutputFailureReturns422WithoutOutput(t *testing.T) {
	h := &openAITestSurface{
		Store: mockOpenAIConfig{},
		Auth:  streamStatusAuthStub{},
		DS: &streamStatusDSSeqStub{resps: []*http.Response{
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":201,"p":"response/content","v":"{\"score\":\"a\"}"}`, `data: [DONE]`),
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":202,"p":"response/content","v":"{\"score\":\"b\"}"}`, `data: [DONE]`),
			makeOpenAISSEHTTPResponse(`data: {"response_message_id":203,"p":"response/content","v":"{\"score\":\"c\"}"}`, `data: [DONE]`),
		}},
	}

	reqBody := `{"model":"deepseek-v4-flash","input":"Return score=9.","stream":false,"text":{"format":{"type":"json_schema","json_schema":{"name":"ScoreCard","strict":true,"schema":{"type":"object","additionalProperties":false,"required":["score"],"properties":{"score":{"type":"integer"}}}}}}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	newOpenAITestRouter(h).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("expected 422, got %d body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if _, ok := body["output"]; ok {
		t.Fatalf("expected fail-closed error body without output, got %s", rec.Body.String())
	}
	if _, ok := body["output_parsed"]; ok {
		t.Fatalf("expected fail-closed error body without output_parsed, got %s", rec.Body.String())
	}
	errObj, _ := body["error"].(map[string]any)
	if errObj["code"] != "structured_output_validation_failed" {
		t.Fatalf("unexpected error code: %#v", errObj)
	}
	msg, _ := errObj["message"].(string)
	if !strings.Contains(msg, "Structured output validation failed") {
		t.Fatalf("unexpected error message: %#v", errObj)
	}
}
