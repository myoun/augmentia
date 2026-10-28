mod analyzer;
mod setup;
mod view;

use analyzer::{DEFAULT_SEED, GameSession};
use serde_json::{Value, json};
use std::error::Error;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

const INDEX_HTML: &str = include_str!("static/index.html");
const APP_JS: &str = include_str!("static/app.js");
const STYLES_CSS: &str = include_str!("static/styles.css");
const MAX_REQUEST_BYTES: usize = 1_048_576;

#[derive(Clone)]
struct Options {
    port: u16,
    seed: u64,
    open_browser: bool,
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_args(std::env::args().skip(1))?;
    let listener = TcpListener::bind(("127.0.0.1", options.port))?;
    let address = listener.local_addr()?;
    let url = format!("http://{address}");
    let session = Arc::new(Mutex::new(GameSession::new(options.seed)?));
    let active_connections = Arc::new(AtomicUsize::new(0));

    println!("증강체스 웹 UI");
    println!("주소: {url}");
    let catalog = augment_core::latest_catalog();
    println!(
        "규칙 스냅숏: {} (카드 {}/{}장 지원) | seed: {}",
        augment_core::LATEST_RULESET_ID,
        catalog.iter().filter(|card| card.supported()).count(),
        catalog.len(),
        options.seed
    );
    println!("종료하려면 Ctrl+C를 누르세요.");
    if options.open_browser {
        let _ = open_browser(&url);
    }

    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                if active_connections.fetch_add(1, Ordering::Relaxed) >= 16 {
                    active_connections.fetch_sub(1, Ordering::Relaxed);
                    let _ = write_json(
                        &mut stream,
                        503,
                        &json!({ "error": "요청이 많습니다. 잠시 후 다시 시도하세요." }),
                    );
                    continue;
                }
                let session = Arc::clone(&session);
                let active = Arc::clone(&active_connections);
                std::thread::spawn(move || {
                    if let Err(error) = handle_connection(&mut stream, &session) {
                        eprintln!("요청 처리 실패: {error}");
                        let _ = write_json(
                            &mut stream,
                            500,
                            &json!({ "error": "서버가 요청을 처리하지 못했습니다." }),
                        );
                    }
                    active.fetch_sub(1, Ordering::Relaxed);
                });
            }
            Err(error) => eprintln!("연결 실패: {error}"),
        }
    }
    Ok(())
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Options, Box<dyn Error>> {
    let mut options = Options {
        port: 7878,
        seed: DEFAULT_SEED,
        open_browser: true,
    };
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--port" => {
                options.port = args
                    .next()
                    .ok_or("--port 뒤에 포트 번호가 필요합니다")?
                    .parse()?;
            }
            "--seed" => {
                options.seed = args
                    .next()
                    .ok_or("--seed 뒤에 정수가 필요합니다")?
                    .parse()?;
            }
            "--no-open" => options.open_browser = false,
            "--help" | "-h" => {
                println!(
                    "사용법: cargo run -p augment-app --bin web -- [--port 7878] [--seed 정수] [--no-open]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("알 수 없는 인자: {argument}").into()),
        }
    }
    Ok(options)
}

fn handle_connection(
    stream: &mut TcpStream,
    shared: &Arc<Mutex<GameSession>>,
) -> Result<(), Box<dyn Error>> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let request = read_request(stream)?;
    // Copy immutable analysis state under the lock, then release it before searching.
    // Navigation and edits remain responsive and cancel the previous search cooperatively.
    if request.method == "POST" && request.path == "/api/analyze" {
        let payload = parse_json_body(&request)?;
        let snapshot = shared
            .lock()
            .map_err(|_| "세션 잠금 실패")?
            .analysis_snapshot();
        return match snapshot.analyze(&payload) {
            Ok(view) => write_json(stream, 200, &view).map_err(Into::into),
            Err(error) => write_json(stream, 400, &json!({ "error": error })).map_err(Into::into),
        };
    }
    let mut session = shared.lock().map_err(|_| "세션 잠금 실패")?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => write_response(
            stream,
            200,
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        ),
        ("GET", "/app.js") => write_response(
            stream,
            200,
            "text/javascript; charset=utf-8",
            APP_JS.as_bytes(),
        ),
        ("GET", "/acg-export.js") => write_response(
            stream,
            200,
            "text/javascript; charset=utf-8",
            include_str!("static/acg-export.js").as_bytes(),
        ),
        ("GET", "/acg-view.js") => write_response(
            stream,
            200,
            "text/javascript; charset=utf-8",
            include_str!("static/acg-view.js").as_bytes(),
        ),
        ("GET", "/acg-codec.js") => write_response(
            stream,
            200,
            "text/javascript; charset=utf-8",
            include_str!("static/acg-codec.js").as_bytes(),
        ),
        ("GET", "/styles.css") => write_response(
            stream,
            200,
            "text/css; charset=utf-8",
            STYLES_CSS.as_bytes(),
        ),
        ("GET", "/favicon.ico") => write_response(stream, 204, "image/x-icon", &[]),
        ("GET", "/api/health") => write_json(stream, 200, &json!({ "ok": true })),
        ("GET", "/api/pieces") => write_json(stream, 200, &setup::piece_catalog()),
        ("GET", "/api/card-catalog") => write_json(
            stream,
            200,
            &json!(
                augment_core::latest_catalog()
                    .iter()
                    .map(|card| view::card_meta(&card.id))
                    .collect::<Vec<_>>()
            ),
        ),
        ("GET", "/api/export-path") => match session.export_path() {
            Ok(path) => write_json(stream, 200, &path),
            Err(error) => write_json(stream, 400, &json!({"error":error})),
        },
        ("POST", "/api/setup-options") => {
            let payload = parse_json_body(&request)?;
            let owner = match payload["owner"].as_str() {
                Some("white") => augment_core::Player::White,
                Some("black") => augment_core::Player::Black,
                _ => {
                    return write_json(stream, 400, &json!({"error":"카드 소유자를 선택하세요."}))
                        .map_err(Into::into);
                }
            };
            write_json(stream, 200, &session.setup_options(owner))
        }
        ("POST", "/api/setup-card") => {
            let payload = parse_json_body(&request)?;
            match session.acquire_setup(&payload) {
                Ok(view) => write_json(stream, 200, &view),
                Err(error) => write_json(stream, 400, &json!({"error":error})),
            }
        }
        ("POST", "/api/edit-board") => {
            let payload = parse_json_body(&request)?;
            match session.edit_board(&payload) {
                Ok(view) => write_json(stream, 200, &view),
                Err(error) => write_json(stream, 400, &json!({"error": error})),
            }
        }
        ("GET", "/api/state") => match session.view() {
            Ok(view) => write_json(stream, 200, &view),
            Err(error) => write_json(stream, 500, &json!({ "error": error })),
        },
        ("POST", "/api/new") => {
            let payload = parse_json_body(&request)?;
            let seed = payload
                .get("seed")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_SEED);
            match session.reset(seed).and_then(|()| session.view()) {
                Ok(view) => write_json(stream, 200, &view),
                Err(error) => write_json(stream, 400, &json!({ "error": error })),
            }
        }
        ("POST", "/api/analysis-stop") => {
            session.stop_analysis();
            write_json(stream, 200, &json!({ "stopped": true }))
        }
        ("POST", "/api/navigate") => {
            let payload = parse_json_body(&request)?;
            let result = payload
                .get("node")
                .and_then(Value::as_u64)
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| "기보 node가 필요합니다.".to_owned())
                .and_then(|id| session.navigate(id));
            match result {
                Ok(view) => write_json(stream, 200, &view),
                Err(error) => write_json(stream, 400, &json!({ "error": error })),
            }
        }
        ("POST", "/api/action") => {
            let payload = parse_json_body(&request)?;
            let Some(index) = payload.get("index").and_then(Value::as_u64) else {
                write_json(stream, 400, &json!({ "error": "행동 index가 필요합니다." }))?;
                return Ok(());
            };
            let fingerprint = payload.get("fingerprint").and_then(Value::as_str);
            match session.apply_index(index as usize, fingerprint) {
                Ok(view) => write_json(stream, 200, &view),
                Err(error) => write_json(stream, 409, &json!({ "error": error })),
            }
        }
        _ => write_json(stream, 404, &json!({ "error": "찾을 수 없는 경로입니다." })),
    }?;
    Ok(())
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, Box<dyn Error>> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err("요청이 헤더 전에 종료되었습니다".into());
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err("요청이 너무 큽니다".into());
        }
        if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = std::str::from_utf8(&bytes[..header_end])?;
    let mut lines = headers.split("\r\n");
    let request_line = lines.next().ok_or("요청 줄이 없습니다")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or("HTTP 메서드가 없습니다")?
        .to_owned();
    let path = request_parts
        .next()
        .ok_or("HTTP 경로가 없습니다")?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if header_end + content_length > MAX_REQUEST_BYTES {
        return Err("요청이 너무 큽니다".into());
    }
    while bytes.len() < header_end + content_length {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err("요청 본문이 일찍 종료되었습니다".into());
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: bytes[header_end..header_end + content_length].to_vec(),
    })
}

fn parse_json_body(request: &HttpRequest) -> Result<Value, Box<dyn Error>> {
    if request.body.is_empty() {
        Ok(json!({}))
    } else {
        Ok(serde_json::from_slice(&request.body)?)
    }
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(value).expect("JSON value serializes");
    write_response(stream, status, "application/json; charset=utf-8", &body)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn()
        .map(|_| ())
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("open")
        .arg(url)
        .spawn()
        .map(|_| ())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("xdg-open")
        .arg(url)
        .spawn()
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_options() {
        let options = parse_args(
            ["--port", "9123", "--seed", "42", "--no-open"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(options.port, 9123);
        assert_eq!(options.seed, 42);
        assert!(!options.open_browser);
    }
}
