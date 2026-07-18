# boxy packaging — build a wheel and push it to your LOCAL (private) PyPI.
# Public-PyPI releases stay tag-driven (see RELEASING.md); this Makefile is the
# short path for an internal index (devpi / Nexus / Artifactory / pypiserver):
#
#     make wheel                                        # dist/boxy_hpc-*.whl
#     make publish LOCAL_PYPI=https://pypi.example/     # build + twine upload
#     make publish LOCAL_PYPI=sandia                    # a [sandia] section in ~/.pypirc
#
# LOCAL_PYPI accepts either an upload URL (goes through --repository-url) or a
# bare ~/.pypirc section name (goes through --repository). It can also come from
# the environment: export LOCAL_PYPI=... once and just `make publish`.
#
# uv-NATIVE: when uv is on PATH everything runs through it — `uv build` needs
# no pip/build in the venv (uv venvs ship WITHOUT pip: `python -m pip` there
# dies with 'No module named pip' — field), and twine runs via `uvx`. Without
# uv, the classic pip/build/twine path is used. Force either with USE_UV=1|0.

PYTHON ?= python3
LOCAL_PYPI ?=
USE_UV ?= $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(USE_UV),1)
BUILD_WHEEL = uv build --wheel
BUILD_SDIST = uv build --sdist
BUILD_ALL   = uv build
TWINE       = uvx twine
DEPS        = @:
else
BUILD_WHEEL = $(PYTHON) -m build --wheel
BUILD_SDIST = $(PYTHON) -m build --sdist
BUILD_ALL   = $(PYTHON) -m build
TWINE       = $(PYTHON) -m twine
DEPS        = $(PYTHON) -m pip install --quiet build twine
endif

.PHONY: deps wheel sdist dist check publish clean

deps:                             ## install the build tools (no-op under uv)
	$(DEPS)

wheel: deps                       ## build only the wheel into dist/
	$(BUILD_WHEEL)

sdist: deps                       ## build only the sdist into dist/
	$(BUILD_SDIST)

dist: deps                        ## build sdist + wheel into dist/
	$(BUILD_ALL)

check: dist                       ## twine-validate the built metadata
	$(TWINE) check dist/*

publish: check                    ## upload dist/* to LOCAL_PYPI (URL or ~/.pypirc section)
	@test -n "$(LOCAL_PYPI)" || { \
	  echo "error: set LOCAL_PYPI to your local PyPI's UPLOAD endpoint or a ~/.pypirc section name:" >&2; \
	  echo "  make publish LOCAL_PYPI=https://pypi.example.gov/          # upload URL" >&2; \
	  echo "  make publish LOCAL_PYPI=sandia                             # [sandia] in ~/.pypirc" >&2; \
	  exit 2; }
	@case "$(LOCAL_PYPI)" in \
	  http://*|https://*) $(TWINE) upload --repository-url "$(LOCAL_PYPI)" dist/* ;; \
	  *)                  $(TWINE) upload --repository "$(LOCAL_PYPI)" dist/* ;; \
	esac

clean:                            ## remove build artifacts
	rm -rf build dist src/*.egg-info
