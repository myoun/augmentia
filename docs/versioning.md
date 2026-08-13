# Git 및 규칙 스냅숏 버전 정책

## 기본 원칙

Git commit은 프로젝트 전체의 변경 이력이고 `ruleset_id`는 한 대국의 실행 의미를
고정한다. 둘은 연결하지만 같은 개념으로 사용하지 않는다.

- `main`: 엔진, 수집기, 학습 코드와 모든 지원 ruleset의 누적 역사
- `rulesets/augment_chess_YYYYMMDD/`: 해당 날짜 웹게임 동작의 불변 스냅숏
- annotated tag `ruleset/augment_chess_YYYYMMDD-rN`: 검증 완료된 스냅숏 릴리스
- engine crate version: 엔진 API/직렬화 호환성 버전
- 모델 checkpoint: ruleset 태그와 학습 코드 commit을 함께 참조하는 외부 산출물

날짜마다 장기 브랜치를 만들지 않는다. 브랜치는 구현 중인 변경을 격리하는 데만 쓰고,
과거 규칙은 디렉터리와 Git 태그로 유지한다. 그래야 최신 엔진에서 여러 과거 스냅숏을
동시에 replay하고 비교할 수 있다.

## 태그 규칙

첫 검증 릴리스는 다음처럼 annotated tag로 고정한다.

```powershell
git tag -a ruleset/augment_chess_20260807-r1 -m "Augment Chess 2026-08-07 ruleset r1"
```

동일 날짜의 자료를 잘못 해석했거나 수집 누락을 정정할 때 기존 태그나 디렉터리를
강제로 바꾸지 않는다. `r2`를 만들고 `supersedes`로 이전 revision을 연결한다.

```text
ruleset/augment_chess_20260807-r1
ruleset/augment_chess_20260807-r2
```

사이트 자체가 변경된 경우에는 revision이 아니라 새 사이트 날짜를 사용한다. 사이트가
업데이트 날짜를 제공하지 않는 무통보 변경이라면 최초 관측 날짜를 사용하고 source
metadata에 `upstream_date_unknown: true`를 남긴다.

현재 `augment_chess_20260807`은 카드 coverage가 `partial`이므로 태그를 만들지 않는다.
태그는 아래 release gate를 모두 통과한 뒤에만 생성한다.

## 웹 번들과 bug compatibility

웹게임 버그도 규칙의 일부다. 각 release revision은 다음 정보를 manifest 또는 source
manifest에 포함해야 한다.

- 실제 HTML에서 발견한 번들 URL
- 번들 전체 SHA-256 및 byte length
- 수집 시각과 upstream 업데이트 날짜
- 원본 193개/기본 공개 192개 같은 필터 결과
- 게임 상태·승패에 영향을 미치는 `known_quirks`
- quirk별 최소 재현 상태, 행동열, 기대 fingerprint

표시 설명과 JS 실행이 다르면 그 revision의 엔진은 JS 실행 결과를 따른다. 의도된 규칙은
`intended_behavior`로 별도 보존한다. 새 사이트 번들이 버그를 수정하면 과거 스냅숏의
golden test를 고치지 않고 새 스냅숏을 만든다.

게임 의미와 무관한 화면·애니메이션·채팅·네트워크 및 벽시계 버그는 게임 엔진 범위에서
제외한다. 단, 해당 동작이 합법 행동, 난수 소비 순서, 보드 상태, 턴 또는 승패 결과를
바꾸면 engine-visible quirk로 포함한다.

## 릴리스 gate

태그를 만들려면 다음 조건을 만족해야 한다.

1. 원본 번들 SHA-256이 기록되고 다시 다운로드해 검증 가능하다.
2. 공개 카드와 RULE의 coverage가 100%다.
3. 모든 effect가 handler와 연결되어 있다.
4. 알려진 게임 로직 버그에 golden regression test가 있다.
5. replay가 동일한 최종 fingerprint를 만든다.
6. formatter, test, clippy가 모두 통과한다.
7. manifest가 현재 Git commit과 release revision을 기록한다.

## 커밋 단위

수집된 원본, 정규화 데이터, 엔진 구현을 가능한 한 분리해 검토한다.

```text
snapshot: capture upstream bundle for 20260807
snapshot: normalize 20260807 card definitions
engine: reproduce parry retaliation ordering
quirk: preserve portal-gun nonce behavior
test: add 20260807 democracy simultaneous-loss fixture
```

원본 snapshot commit은 다운로드한 자료와 digest만 포함한다. 의미 해석과 엔진 코드는
후속 commit에 둬서 나중에 추출 오류와 엔진 오류를 구분할 수 있게 한다.

## 학습 산출물

대용량 checkpoint와 replay buffer는 일반 Git에 넣지 않는다. 기본적으로 로컬/객체
스토리지에 저장하고 다음 provenance JSON만 Git에 커밋한다.

```json
{
  "ruleset_tag": "ruleset/augment_chess_20260807-r1",
  "engine_commit": "<40-character commit>",
  "training_commit": "<40-character commit>",
  "seed": 20260807,
  "artifact_sha256": "<sha256>",
  "artifact_uri": "<object-store-or-release-uri>"
}
```

필요하면 작은 baseline 모델만 Git LFS로 추적한다. LFS 사용 여부는 저장소 원격과 배포
방식이 정해진 뒤 결정하며, 현재는 `.gitattributes`에 바이너리 추적 규칙을 넣지 않는다.
