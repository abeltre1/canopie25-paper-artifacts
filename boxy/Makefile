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

PYTHON ?= python3
LOCAL_PYPI ?=

.PHONY: deps wheel sdist dist check publish clean

deps:                             ## install the build tools (build + twine)
	$(PYTHON) -m pip install --quiet build twine

wheel: deps                       ## build only the wheel into dist/
	$(PYTHON) -m build --wheel

sdist: deps                       ## build only the sdist into dist/
	$(PYTHON) -m build --sdist

dist: deps                        ## build sdist + wheel into dist/
	$(PYTHON) -m build

check: dist                       ## twine-validate the built metadata
	$(PYTHON) -m twine check dist/*

publish: check                    ## upload dist/* to LOCAL_PYPI (URL or ~/.pypirc section)
	@test -n "$(LOCAL_PYPI)" || { \
	  echo "error: set LOCAL_PYPI to your local PyPI's UPLOAD endpoint or a ~/.pypirc section name:" >&2; \
	  echo "  make publish LOCAL_PYPI=https://pypi.example.gov/          # upload URL" >&2; \
	  echo "  make publish LOCAL_PYPI=sandia                             # [sandia] in ~/.pypirc" >&2; \
	  exit 2; }
	@case "$(LOCAL_PYPI)" in \
	  http://*|https://*) $(PYTHON) -m twine upload --repository-url "$(LOCAL_PYPI)" dist/* ;; \
	  *)                  $(PYTHON) -m twine upload --repository "$(LOCAL_PYPI)" dist/* ;; \
	esac

clean:                            ## remove build artifacts
	rm -rf build dist src/*.egg-info
