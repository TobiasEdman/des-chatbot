# Cross-runtime coordination

This directory is the vendor-neutral coordination substrate for agent runtimes
and human contributors.

## Required workflow

1. Create an exclusively owned worktree and runtime-labelled branch.

2. Provision the exact published operator wheel into a dedicated external
   virtual environment. The target repository never supplies executable
   verifier or provisioning code. Obtain an authenticated release bundle
   containing the wheel, canonical detached receipt, and standalone
   verified_toolchain.py. Keep the release root, reviewed source checkout,
   runtime root, operator environment, governance credential root, and every
   target repository mutually disjoint. The copy of this README and the
   repository toolchain lock are data, never release authority.

   Select the complete profile for the exact OS, architecture, and Python ABI.
   A profile binds runtime-root-relative paths and digests for base Python, its
   native runtime and stdlib, Docker, Git, the hash utility, the provisioner, dependency
   wheel artifacts and installed trees, package metadata, the immutable source
   commit, uv.lock, and the final wheel. Never discover any of those tools
   through caller PATH.

   The following is the exact command shape. Replace each /absolute placeholder
   only with paths delivered by the authenticated profile. OPERATOR_SITE is the
   profile's X.Y site-packages path; do not compute it by executing the new
   virtual environment.

   ~~~bash
   REPOSITORY=/absolute/target-repository
   REVIEWED_SOURCE=/absolute/reviewed/multi-agentic-at-receipt-commit
   OPERATOR_RELEASE_ROOT=/absolute/authenticated-release
   OPERATOR_RUNTIME_ROOT=/absolute/reviewed-runtime
   OPERATOR_PREFIX=/absolute/dedicated-operator-venv
   OPERATOR_WHEEL="$OPERATOR_RELEASE_ROOT/multi_agentic-0.7.0-py3-none-any.whl"
   RELEASE_LOCK="$OPERATOR_RELEASE_ROOT/toolchain-lock.json"
   OPERATOR_PROFILE=Darwin-arm64-cpython-314
   INSTALL_PROFILE=coordination

   case "$OPERATOR_PROFILE" in
     Darwin-arm64-cpython-314)
       OPERATOR_BASE_PREFIX="$OPERATOR_RUNTIME_ROOT/cpython-3.14.3-macos-aarch64-none"
       OPERATOR_PYTHON="$OPERATOR_RUNTIME_ROOT/cpython-3.14.3-macos-aarch64-none/bin/python3.14"
       OPERATOR_RUNTIME_LIBRARY="$OPERATOR_BASE_PREFIX/lib/libpython3.14.dylib"
       OPERATOR_DOCKER="$OPERATOR_RUNTIME_ROOT/tools/docker"
       OPERATOR_GIT="$OPERATOR_RUNTIME_ROOT/tools/git"
       OPERATOR_HASH="$OPERATOR_RUNTIME_ROOT/tools/hash"
       OPERATOR_UV="$OPERATOR_RUNTIME_ROOT/tools/uv"
       OPERATOR_SITE="$OPERATOR_PREFIX/lib/python3.14/site-packages"
       OPERATOR_VENV_RUNTIME_LIBRARY="$OPERATOR_PREFIX/lib/libpython3.14.dylib"
       ;;
     Linux-aarch64-cpython-312)
       OPERATOR_BASE_PREFIX="$OPERATOR_RUNTIME_ROOT/local"
       OPERATOR_PYTHON="$OPERATOR_RUNTIME_ROOT/local/bin/python3.12"
       OPERATOR_RUNTIME_LIBRARY="$OPERATOR_BASE_PREFIX/lib/libpython3.12.so.1.0"
       OPERATOR_DOCKER="$OPERATOR_RUNTIME_ROOT/bin/docker"
       OPERATOR_GIT="$OPERATOR_RUNTIME_ROOT/bin/git"
       OPERATOR_HASH="$OPERATOR_RUNTIME_ROOT/bin/sha256sum"
       OPERATOR_UV="$OPERATOR_RUNTIME_ROOT/local/bin/uv"
       OPERATOR_SITE="$OPERATOR_PREFIX/lib/python3.12/site-packages"
       OPERATOR_VENV_RUNTIME_LIBRARY="$OPERATOR_PREFIX/lib/libpython3.12.so.1.0"
       ;;
     *) exit 64 ;;
   esac
   case "$INSTALL_PROFILE" in
     coordination) PROFILE_EXTRA_ARGS=() ;;
     operator) PROFILE_EXTRA_ARGS=(--extra operator) ;;
     service|worker) PROFILE_EXTRA_ARGS=(--extra service) ;;
     *) exit 64 ;;
   esac
   OPERATOR_VERIFIER="$OPERATOR_SITE/agentic_task/verified_toolchain.py"

   "$OPERATOR_PYTHON" -I -S -B \
     "$OPERATOR_RELEASE_ROOT/verified_toolchain.py" preflight \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" \
     --install-profile "$INSTALL_PROFILE" \
     --lock "$RELEASE_LOCK" --python "$OPERATOR_PYTHON" \
     --git "$OPERATOR_GIT" --docker "$OPERATOR_DOCKER" \
     --hash-tool "$OPERATOR_HASH" \
     --provisioner "$OPERATOR_UV" --source "$REVIEWED_SOURCE" \
     --wheel "$OPERATOR_WHEEL"

   OPERATOR_WHEEL_SHA256="$(
     "$OPERATOR_PYTHON" -I -S -B -c \
       'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["package"]["wheel"]["sha256"],end="")' \
       "$RELEASE_LOCK"
   )"
   OPERATOR_WHEEL_REQUIREMENT="multi-agentic @ file://$OPERATOR_WHEEL#sha256=$OPERATOR_WHEEL_SHA256"

   "$OPERATOR_PYTHON" -I -S -B -m venv \
     --copies --without-pip "$OPERATOR_PREFIX"
   "$OPERATOR_PYTHON" -I -S -B -c \
     'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])' \
     "$OPERATOR_RUNTIME_LIBRARY" "$OPERATOR_VENV_RUNTIME_LIBRARY"
   "$OPERATOR_PYTHON" -I -S -B -c \
     'import os,sys; root=sys.argv[1]; names=("activate","activate.csh","activate.fish","Activate.ps1"); [os.unlink(os.path.join(root,n)) for n in names if os.path.lexists(os.path.join(root,n))]' \
     "$OPERATOR_PREFIX/bin"
   "$OPERATOR_UV" --directory "$REVIEWED_SOURCE" export --locked --no-dev \
     "${PROFILE_EXTRA_ARGS[@]}" \
     --no-emit-project --output-file "$OPERATOR_PREFIX/requirements.lock"
   UV_LINK_MODE=copy "$OPERATOR_UV" pip install \
     --python "$OPERATOR_PREFIX/bin/python" --require-hashes --no-deps \
     -r "$OPERATOR_PREFIX/requirements.lock"
   UV_LINK_MODE=copy "$OPERATOR_UV" pip install \
     --python "$OPERATOR_PREFIX/bin/python" --no-deps \
     "$OPERATOR_WHEEL_REQUIREMENT"
   "$OPERATOR_PYTHON" -I -S -B -c \
     'import os,sys; os.unlink(sys.argv[1])' \
     "$OPERATOR_VENV_RUNTIME_LIBRARY"

   "$OPERATOR_PYTHON" -I -S -B \
     -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
     "$OPERATOR_VERIFIER" \
     --repo "$REPOSITORY" --operator-prefix "$OPERATOR_PREFIX" \
     --install-profile "$INSTALL_PROFILE" \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" --git "$OPERATOR_GIT" \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --wheel "$OPERATOR_WHEEL" --release-lock "$RELEASE_LOCK" \
     --author-name "${AGENT_AUTHOR_NAME:?this runtime's agent author}" --author-email "${AGENT_AUTHOR_EMAIL:?this runtime's vendor noreply address}" \
     verify
   ~~~

   That block provisions and verifies exactly one selected install profile.
   Repeat it with a distinct `OPERATOR_PREFIX` and matching `OPERATOR_SITE` for
   each needed profile. Only the coordination profile performs repository
   bootstrap; after its `verify` succeeds, run this separate exact command:

   ~~~bash
   "$OPERATOR_PYTHON" -I -S -B \
     -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
     "$OPERATOR_VERIFIER" \
     --repo "$REPOSITORY" --operator-prefix "$OPERATOR_PREFIX" \
     --install-profile coordination \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" --git "$OPERATOR_GIT" \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --wheel "$OPERATOR_WHEEL" --release-lock "$RELEASE_LOCK" \
     --author-name "${AGENT_AUTHOR_NAME:?this runtime's agent author}" --author-email "${AGENT_AUTHOR_EMAIL:?this runtime's vendor noreply address}" \
     agentic-task -- init "$REPOSITORY" --profile coordination --apply
   ~~~

   Never source or retain activation helpers, and never invoke an executable in
   the operator environment directly. Every preflight and launcher invocation
   runs through the profile-bound base Python. The copied environment is an
   import tree inspected by the launcher. Lock-free operation is limited to the
   exact init --apply grammar with the authenticated detached receipt. An
   existing different lock is never replaced. After bootstrap, the repository
   copy must remain byte-identical to the detached receipt.

   Provision each advertised install profile into a distinct prefix. The
   coordination prefix enables only agentic-task and agentic-continuity; the
   operator prefix enables only policy verification, operator signing, and the
   host active-binding command; the worker prefix enables only the host worker;
   the service profile is confined to the immutable service image. A launcher
   command is refused if its `--install-profile` does not match this matrix.

   Editable, symlinked, hard-linked, same-version-but-different, unrecorded,
   cached-bytecode, .pth, sitecustomize, extra wheel payload, and unexpected
   installer metadata are rejected. UV_LINK_MODE=copy is mandatory. -B prevents
   bytecode writes; it does not authenticate code that already ran. The base
   runtime and OS are explicit external trust roots before Python starts, while
   the scanner rejects cached bytecode in the inspected installed tree. A
   compromised OS, administrator, or same-UID operator account is outside this
   boundary.

   Repository scripts, ambient commands, caller Git variables, global/system
   Git config, hooks, attributes, fsmonitor, signing, replacement objects, URL
   rewrites, and caller PATH are untrusted. The author name, author email,
   AGENT_ID, and AGENT_RUNTIME values are validated attribution assertions, not
   authenticated identities. The verified launcher accepts no Git secret and
   performs no private-Git coordination write. Autonomous coordination uses the
   locked HTTPS governance service and GitHub App. Candidate branch publication
   is a separate writer action whose exact SHA is checked later.

   Published profiles are Darwin-arm64-cpython-314 and
   Linux-aarch64-cpython-312. Windows, x86_64, other Python ABIs, and WSL that
   does not exactly match a published Linux profile fail closed. This contract
   requires POSIX no-follow descriptor support. SSH, local, extension, and
   plaintext Git transports are refused. The verified launcher also refuses
   dryrun because its test command executes candidate shell code; that work
   belongs only in the credential-free isolated worker profile.

3. Configure service credentials outside every repository, Git, release,
   runtime, and operator root. The credential root is current-owner mode 0700.
   Its exact canonical HTTPS origin lock and bearer token are distinct direct
   children; the token is mode 0600. Pass all bindings explicitly on every
   service-backed command:

   ~~~bash
   GOVERNANCE_URL=https://governance.example.invalid
   GOVERNANCE_CREDENTIAL_ROOT=/absolute/governance-credentials
   GOVERNANCE_ORIGIN_FILE="$GOVERNANCE_CREDENTIAL_ROOT/origin.txt"
   GOVERNANCE_TOKEN_FILE="$GOVERNANCE_CREDENTIAL_ROOT/operator.token"

   "$OPERATOR_PYTHON" -I -S -B \
     -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
     "$OPERATOR_VERIFIER" \
     --repo "$REPOSITORY" --operator-prefix "$OPERATOR_PREFIX" \
     --install-profile coordination \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" --git "$OPERATOR_GIT" \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --release-lock "$RELEASE_LOCK" --wheel "$OPERATOR_WHEEL" \
     --author-name "${AGENT_AUTHOR_NAME:?this runtime's agent author}" --author-email "${AGENT_AUTHOR_EMAIL:?this runtime's vendor noreply address}" \
     --governance-url "$GOVERNANCE_URL" \
     --governance-credential-root "$GOVERNANCE_CREDENTIAL_ROOT" \
     --governance-origin-file "$GOVERNANCE_ORIGIN_FILE" \
     --governance-token-file "$GOVERNANCE_TOKEN_FILE" \
     agentic-task -- governance session-start "$REPOSITORY" \
     --session session-uuid --clone-id clone-uuid \
     --worktree-id worktree-uuid --role writer \
     --runtime codex --model exact-model-id
   ~~~

   The launcher descriptor-opens the root and files, checks ownership, exact
   modes, ACL absence, link count, content, and identity, and carries bindings
   that the child rechecks before local reads and HTTP mutations.

4. Register the host proposer latch only through the operator profile. The
   operator must pre-create `OPERATOR_DATA_ROOT` and its `active-bindings`
   child as distinct, current-owner mode 0700 roots outside the repository,
   Git, release, runtime, venv, and governance credential roots. The output
   name must not exist. This is a service command and therefore uses the exact
   URL, origin lock, and token binding from step 3:

   ~~~bash
   OPERATOR_DATA_ROOT=/absolute/operator-data
   ACTIVE_BINDING_REQUEST_ID=active-binding-request-uuid
   ACTIVE_BINDING_NONCE=replace-with-unique-receipt-nonce
   ACTIVE_BINDING_OUTPUT="$OPERATOR_DATA_ROOT/active-bindings/robotics-stack-$ACTIVE_BINDING_NONCE.json"

   "$OPERATOR_PYTHON" -I -S -B -c 'if True:
   import os
   import stat
   import sys

   root = os.path.abspath(sys.argv[1])
   output = os.path.abspath(sys.argv[2])
   flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
   root_fd = os.open(root, flags)
   try:
       root_stat = os.fstat(root_fd)
       if (
           not stat.S_ISDIR(root_stat.st_mode)
           or root_stat.st_uid != os.getuid()
           or stat.S_IMODE(root_stat.st_mode) != 0o700
       ):
           raise SystemExit("operator data root must be current-owner mode 0700")
       try:
           os.mkdir("active-bindings", 0o700, dir_fd=root_fd)
       except FileExistsError:
           pass
       binding_fd = os.open("active-bindings", flags, dir_fd=root_fd)
       try:
           binding_stat = os.fstat(binding_fd)
           if (
               not stat.S_ISDIR(binding_stat.st_mode)
               or binding_stat.st_uid != os.getuid()
               or stat.S_IMODE(binding_stat.st_mode) != 0o700
           ):
               raise SystemExit("active-binding root must be current-owner mode 0700")
           if os.path.dirname(output) != os.path.join(root, "active-bindings"):
               raise SystemExit("active-binding output must be a direct child")
           try:
               os.stat(os.path.basename(output), dir_fd=binding_fd, follow_symlinks=False)
           except FileNotFoundError:
               pass
           else:
               raise SystemExit("active-binding output must not already exist")
       finally:
           os.close(binding_fd)
   finally:
       os.close(root_fd)
   ' "$OPERATOR_DATA_ROOT" "$ACTIVE_BINDING_OUTPUT"

   "$OPERATOR_PYTHON" -I -S -B \
     -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
     "$OPERATOR_VERIFIER" \
     --repo "$REPOSITORY" --operator-prefix "$OPERATOR_PREFIX" \
     --install-profile operator \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" --git "$OPERATOR_GIT" \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --release-lock "$RELEASE_LOCK" --wheel "$OPERATOR_WHEEL" \
     --author-name "${AGENT_AUTHOR_NAME:?this runtime's agent author}" --author-email "${AGENT_AUTHOR_EMAIL:?this runtime's vendor noreply address}" \
     --governance-url "$GOVERNANCE_URL" \
     --governance-credential-root "$GOVERNANCE_CREDENTIAL_ROOT" \
     --governance-origin-file "$GOVERNANCE_ORIGIN_FILE" \
     --governance-token-file "$GOVERNANCE_TOKEN_FILE" \
     --operator-data-root "$OPERATOR_DATA_ROOT" \
     agentic-active-binding -- \
     --target-branch main --request-id "$ACTIVE_BINDING_REQUEST_ID" \
     --allowed-read-tool Read --allowed-read-tool Glob \
     --allowed-read-tool Grep --allowed-read-tool view_image \
     --allowed-read-tool update_plan --output "$ACTIVE_BINDING_OUTPUT"
   ~~~

   The wheel-owned command derives the repository, reviewed Git, and prewrite
   module from the verified context. It sends the unsigned canonical binding
   to the governance service and writes only the returned signed receipt. The
   active proposer guard denies every Bash invocation; no direct console script
   or shell broker gains authority. A repository utility or Compose container
   is not active-binding authority.

5. Add a complete pending task and GoalContract to protected base in Phase A.
   In Phase B, claim its exact path scope through governance claim using the
   identical full launcher and service prefix from step 3. The response's
   fencing token is required for governance write, renew, handoff, and release.
   Never run an installed console script directly.

6. Coordinate only the exact claimed paths. Before each write, obtain service
   authorization with the current fencing token. On transfer, publish a service
   handoff. Committed immutable handoff JSON remains useful continuity evidence,
   but it grants no authority.

7. Submit the exact candidate through submit-prepare and submit, accept the
   service-selected cross-vendor assignment, record the structured review, and
   request exact-SHA merge authorization. Assignment credential output must be
   one new direct child of the bound governance credential root. The GitHub App
   performs the merge compare-and-swap. Durable service and repository records
   remain authoritative after every local session exits.

8. Dispatch protected-policy verification only from the reviewed wheel through
   the same non-service launcher. Its key files are direct children of the
   release trust root; output is a direct child of a distinct owner-only output
   root:

   ~~~bash
   "$OPERATOR_PYTHON" -I -S -B \
     -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
     "$OPERATOR_VERIFIER" \
     --repo "$REPOSITORY" --operator-prefix "$OPERATOR_PREFIX" \
     --install-profile operator \
     --runtime-root "$OPERATOR_RUNTIME_ROOT" --git "$OPERATOR_GIT" \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --release-lock "$RELEASE_LOCK" --wheel "$OPERATOR_WHEEL" \
     --author-name "${AGENT_AUTHOR_NAME:?this runtime's agent author}" --author-email "${AGENT_AUTHOR_EMAIL:?this runtime's vendor noreply address}" \
     agentic-policy-verify -- \
     --trust-root "$OPERATOR_RELEASE_ROOT" \
     --key-file "$OPERATOR_RELEASE_ROOT/policy-evidence.key" \
     --github-app-private-key-file "$OPERATOR_RELEASE_ROOT/policy-app.pem" \
     --output-root /absolute/policy-output \
     --output /absolute/policy-output/protected-policy-bundle.json \
     --repository approved-owner/robotics-stack \
     --github-app-id 1 --github-installation-id 2 --integration-app-id 3
   ~~~

## Release engineering

Release publication is a two-stage, review-gated operation. First, a release
engineer generates the embedded template for every advertised
OS/architecture/ABI and install-profile pair from a clean, non-editable venv.
Each generation is chained from the previous canonical template; it records
the exact selected wheel artifacts and installed trees. The final template is
committed with all detached fields still zero. An independent reviewer checks
that commit, the four profile inventories, the two supported platform
inventories, and the locked `uv.lock` before approving the release commit.

For each clean profile environment, chain the prior template output into the
next invocation of the deterministic generator:

~~~bash
PROFILE_TEMPLATE_INPUT=/absolute/external/previous-toolchain-template.json
PROFILE_TEMPLATE_OUTPUT=/absolute/external/next-toolchain-template.json
PROFILE_WHEEL=/absolute/external/provisional/multi_agentic-0.7.0-py3-none-any.whl

"$OPERATOR_PYTHON" -I -S -B \
  -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
  "$REVIEWED_SOURCE/src/agentic_task/verified_toolchain.py" generate-lock \
  --template "$PROFILE_TEMPLATE_INPUT" --wheel "$PROFILE_WHEEL" \
  --uv-lock "$REVIEWED_SOURCE/uv.lock" \
  --source-root "$REVIEWED_SOURCE" \
  --runtime-root "$OPERATOR_RUNTIME_ROOT" \
  --install-profile "$INSTALL_PROFILE" \
  --git "$OPERATOR_GIT" --docker "$OPERATOR_DOCKER" \
  --hash-tool "$OPERATOR_HASH" --provisioner "$OPERATOR_UV" \
  --site-root "$OPERATOR_SITE" --output "$PROFILE_TEMPLATE_OUTPUT" \
  --mode template
~~~

The release engineer copies only the final canonical template to the source
resource, obtains review, and creates the immutable release commit. The
generator derives the checkout HEAD and rejects every dirty path except the
template during this fixed-point stage; it never accepts a caller-asserted SHA.

After that package freeze, use the exact clean commit and run this command once
for each of `coordination`, `operator`, `worker`, and `service`, with that
profile's distinct `OPERATOR_PREFIX` and `OPERATOR_SITE`. No command below may
be discovered through `PATH`.

~~~bash
RELEASE_SOURCE_DATE_EPOCH=1767225600
RELEASE_BUILD_CACHE=/absolute/external/release-build-cache
RELEASE_OUTPUT_PARENT=/absolute/external/release-candidates
INSTALL_PROFILE=coordination
RELEASE_OUTPUT_ROOT="$RELEASE_OUTPUT_PARENT/$OPERATOR_PROFILE-$INSTALL_PROFILE"

"$OPERATOR_PYTHON" -I -S -B \
  -X pycache_prefix=/dev/null/agentic-toolchain-pycache-disabled \
  "$REVIEWED_SOURCE/src/agentic_task/verified_toolchain.py" release-bundle \
  --template "$REVIEWED_SOURCE/src/agentic_task/resources/toolchain-lock.json" \
  --uv-lock "$REVIEWED_SOURCE/uv.lock" \
  --source-root "$REVIEWED_SOURCE" \
  --runtime-root "$OPERATOR_RUNTIME_ROOT" \
  --install-profile "$INSTALL_PROFILE" \
  --python "$OPERATOR_PYTHON" --git "$OPERATOR_GIT" \
  --docker "$OPERATOR_DOCKER" --hash-tool "$OPERATOR_HASH" \
  --provisioner "$OPERATOR_UV" --site-root "$OPERATOR_SITE" \
  --build-cache-root "$RELEASE_BUILD_CACHE" \
  --output-root "$RELEASE_OUTPUT_ROOT" \
  --source-date-epoch "$RELEASE_SOURCE_DATE_EPOCH"
~~~

`release-bundle` derives a clean exact source commit, builds the normal wheel
twice with the fixed epoch and offline cache, requires byte identity, generates
the detached canonical receipt, extracts the standalone preflight from that
wheel, and black-box verifies the selected non-editable profile. It publishes
only by an atomic rename into a previously absent output root. Run all eight
platform/profile combinations in their reviewed release environments before
publication and attach the complete command logs to the durable release
record. The repository CI does not provision these external trust roots and is
not evidence for this release prerequisite. The base-Python comparison below
must report no difference between every candidate's wheel, receipt, and
preflight before one candidate is promoted:

~~~bash
"$OPERATOR_PYTHON" -I -S -B -c \
  'import pathlib,sys; roots=[pathlib.Path(p) for p in sys.argv[1:]]; names=("multi_agentic-0.7.0-py3-none-any.whl","toolchain-lock.json","verified_toolchain.py"); assert roots and all((root/name).read_bytes()==(roots[0]/name).read_bytes() for root in roots[1:] for name in names)' \
  "$RELEASE_OUTPUT_PARENT/Darwin-arm64-cpython-314-coordination" \
  "$RELEASE_OUTPUT_PARENT/Darwin-arm64-cpython-314-operator" \
  "$RELEASE_OUTPUT_PARENT/Darwin-arm64-cpython-314-worker" \
  "$RELEASE_OUTPUT_PARENT/Darwin-arm64-cpython-314-service" \
  "$RELEASE_OUTPUT_PARENT/Linux-aarch64-cpython-312-coordination" \
  "$RELEASE_OUTPUT_PARENT/Linux-aarch64-cpython-312-operator" \
  "$RELEASE_OUTPUT_PARENT/Linux-aarch64-cpython-312-worker" \
  "$RELEASE_OUTPUT_PARENT/Linux-aarch64-cpython-312-service"
~~~

The human release operator is the publishing authority and places the reviewed
bundle in the owner-only external release root. The independent reviewer signs
off on the printed commit and wheel SHA-256 in the durable release record.
There is currently no repository-supplied code-signing authority: do not treat
an unauthenticated download, repository copy, or matching package version as a
release. A collaborator reconstructs trust only from that operator-controlled
bundle and its canonical detached receipt.

## Structured shared goals and autonomous merge

Autonomous merge is available only for a task whose complete GoalContract was
already committed on the exact protected base before the task was claimed. A
minimal structured task adds goal with schema_version, objective, exact scope,
acceptance_criteria, non_goals, and constraints, plus the matching
goal_sha256. Goal strings must be trimmed NFC text without control, format,
bidi, private, surrogate, or unassigned Unicode characters. Digesting uses
UTF-8 JSON with sorted keys and compact separators.

Task creation and implementation are distinct phases. Phase A proposes only one
new pending task/GoalContract for protected-base integration; it must not
include code or policy changes. Phase B claims that now-canonical task and may
change only its validated lifecycle/ownership fields plus the exact
implementation scope. A task created or goal rewritten on the feature branch
is terminally routed to human bootstrap, never treated as autonomous evidence.

For Phase B, submit-prepare returns a short-lived single-use challenge binding
the writer principal/session/model/vendor, task, exact head/base/target,
GoalContract digest, protected policy commit and hash, pull request, and
required check/App set. Submit must explicitly echo the challenge and goal
digest. The service recomputes every trusted field before consuming it.
Cooperative local acknowledgement metadata grants no merge authority.

Review is assigned to an authenticated model from a different verified vendor
than the writer and every code actor. Its structured PASS binds the exact goal,
reviewer-system policy, candidate SHA, protected policy, and CI set. Any goal,
SHA, target, policy, required-check/App, or result change invalidates prior
acknowledgement and review. The GitHub App revalidates those bindings at the
single-use merge compare-and-swap.

Ambiguity, reviewer disagreement, security/legal work, public publication,
destructive/revert work, or protected control-plane changes become durable,
monotonic human_escalation_required records before review assignment. Agents
cannot revise or adjudicate around them.

## Service authority and recovery

The authenticated governance service is authoritative for sessions, path
leases, fencing tokens, handoffs, assignments, reviews, approvals, and merge
records. Its GitHub App is the only autonomous integration writer. Older local
claim-ref commands remain migration compatibility code, are not autonomous
authority, and are refused by the verified launcher. A repository remote URL,
local task JSON, attribution flags, or inbox message authenticates no principal
and grants no write or merge capability.

If a service mutation is rejected or its response is lost, query the service
audit and exact task/session state with the same verified identity. Idempotency
keys, leases, fencing tokens, and exact-SHA records make recovery explicit.
Never infer success from a local commit, reset or rewrite a caller's worktree,
or bypass a stale fencing token.

The compatibility file lifecycle remains pending to claimed to completed, with
in_progress as the claimed intermediate state. That metadata is continuity
data; the service audit and exact-SHA merge records are authoritative.

schema.json defines the task format; handoff-schema.json defines the
committable cross-runtime handoff. install-manifest.json records only artifacts
managed or adopted by the installer so rollback stays narrow and drift-safe.
toolchain-lock.json binds the reviewed external wheel runtime; the installed
agentic_task.verified_toolchain enforces it without loading executable code
from the repository. Personal checkpoints, mechanical snapshots, the live
inbox, active-session state, and active-job locks remain outside Git under
~/.agents/continuity/.
