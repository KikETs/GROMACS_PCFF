# `stale_host_reports`

이 디렉터리는 active host report inventory에 더 이상 섞이면 안 되는 예전 수집물이나
mixed-era JSON을 격리하기 위한 자리다.

원칙:

- `host_reports/`에는 fresh current-semantics report만 둔다.
- stale 또는 legacy report는 aggregate 입력으로 쓰지 않는다.
- 과거 JSON을 보관해야 한다면 이 디렉터리로 옮기고, final claim evidence에서 제외한다.

현재 저장소에서는 stale generic report 파일을 active inventory에서 제거했다.
