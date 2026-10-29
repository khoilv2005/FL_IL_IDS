# NOVUS — kiểm tra ngày 26/10/2026

Project: E:\Study\SE373\Project\vuongdat67\novus

## Đã chạy

- Tạo .venv Python 3.12.7, cài editable project và dependencies [dev] bằng uv.
- CLI `python -m agentic_pentest.presentation.cli --help`: chạy thành công.
- `pytest -q tests/unit --cov=agentic_pentest`: **209 passed**, 96.61 giây.
- Coverage: 68% (9.582 statements; 3.089 chưa được phủ).
- `ruff check . --no-cache`: đạt.
- `ruff format --check . --no-cache`: 247 file đúng định dạng.
- `mypy --strict src`: đạt, 114 source files.
- Server demo chạy trên loopback với port tự cấp, sau kiểm thử đã đóng.
- HTTP health = 200. Differential BOLA verifier xác nhận bộ ba 200/200/401.
- Benchmark demo: **35/41**, EX = 85,37%. Không dùng LLM/API key.

## Benchmark chưa đạt

- EVAL-11: tìm thấy 0 endpoint SPA; server demo là REST mock, không cung cấp SPA.
- EVAL-24, EVAL-31, EVAL-33, EVAL-34: bị scope/executor rate limiter chặn do vượt
  giới hạn 10 request/giây trong lượt benchmark; không tắt giới hạn này.
- EVAL-38: điều kiện kiểm tra prompt-injection sanitizer không đạt.

EX không phải accuracy của model NVIDIA NIM. Benchmark dispatch theo ID câu hỏi,
không gọi model để thực hiện từng câu; một số câu trả True cố định (ví dụ EVAL-05,
EVAL-10, EVAL-28, EVAL-29, EVAL-30, EVAL-41). Không dùng điểm này để tuyên bố chất
lượng suy luận LLM hoặc zero false positives.

## Chưa chạy

- NVIDIA NIM thật: chờ tên model và key của người dùng.
- crAPI thật: Docker executable có sẵn nhưng Linux engine chưa hoạt động.
- Go fastprobe thật: chưa có Go trong PATH.
- Browser crawler thực tế: package Playwright đã cài; chưa cài/chạy browser engine.

## Chuẩn bị NVIDIA NIM

Adapter hiện tại hỗ trợ OpenAI-compatible endpoint. Hosted NVIDIA endpoint:
https://integrate.api.nvidia.com/v1
Tài liệu: https://docs.api.nvidia.com/nim/reference/llm-apis

Khi có tên model, đặt trong `.env` tại root project:

```dotenv
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=<exact model ID supplied by user>
OPENAI_API_KEY=<NVIDIA API key>
LLM_PROVIDER=openai
```

Không ghi key vào mã nguồn, báo cáo hoặc commit. `.env` đã nằm trong .gitignore.

Lưu ý adapter `_call_openai_native` có fallback tự chuyển sang model khác khi lỗi
model không tồn tại/deprecated; CLI khởi tạo LiteLLMClient(fallback_offline=True).
Do đó bước test NIM tiếp theo phải xác minh đúng model, không coi heuristic offline
hoặc model fallback là lần gọi model được yêu cầu thành công.

## Chạy CLI

```powershell
cd E:\Study\SE373\Project\vuongdat67\novus
.\.venv\Scripts\python.exe -m agentic_pentest.presentation.cli --help
```

Khi muốn chạy demo tương tác, mở terminal 1:

```powershell
.\.venv\Scripts\python.exe -m agentic_pentest.presentation.cli demo-server --host 127.0.0.1 --port 8888
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m agentic_pentest.presentation.cli chat --target http://127.0.0.1:8888
```

## Báo cáo máy đọc được

- reports/codex-test/pytest.xml
- reports/codex-test/coverage.xml
- reports/codex-test/demo-benchmark.json

Không sửa source của novus trong lần kiểm tra này.
