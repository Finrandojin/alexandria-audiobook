"""Standalone tests for extract_epub_text (app/app.py).

Builds small EPUB fixtures programmatically with `zipfile` in a temp
directory and exercises href resolution (percent-encoding, '../'
traversal, case variants), paragraph-structure preservation, EPUB3 nav
doc skipping, and the ValueError raised for an unresolvable href.

Run directly:
    python app/test_epub_extract.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import os
import re
import sys
import types
import zipfile
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# app.py pulls in `project` (-> tts.py -> numpy/torch/etc.) purely for
# unrelated TTS/project-management functionality that this test does not
# exercise. Those heavy, possibly-uninstalled dependencies have nothing to
# do with EPUB text extraction, so stub the `project` module in sys.modules
# before importing app.py to keep this test lightweight and standalone.
if 'project' not in sys.modules:
    _fake_project = types.ModuleType('project')

    class _FakeProjectManager:
        def __init__(self, *args, **kwargs):
            pass

        def load_chunks(self):
            return []

        def save_chunks(self, chunks):
            pass

        def __getattr__(self, name):
            # Any other attribute app.py's module-level code happens to
            # touch becomes a harmless no-op callable.
            def _noop(*args, **kwargs):
                return None
            return _noop

    _fake_project.ProjectManager = _FakeProjectManager
    sys.modules['project'] = _fake_project

from app import extract_epub_text  # noqa: E402


CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def make_epub(path, opf_xml, files):
    """Build a minimal EPUB zip at `path`.

    files: dict of archive-relative-path -> str content (written under
    OEBPS/ except for META-INF/container.xml and mimetype).
    """
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('mimetype', 'application/epub+zip')
        zf.writestr('META-INF/container.xml', CONTAINER_XML)
        zf.writestr('OEBPS/content.opf', opf_xml)
        for name, content in files.items():
            zf.writestr(name, content)


def opf(manifest_items, spine_items):
    """Build a minimal OPF XML string.

    manifest_items: list of (id, href, media_type, properties_or_None)
    spine_items: list of idref strings
    """
    items = []
    for item_id, href, media_type, props in manifest_items:
        props_attr = f' properties="{props}"' if props else ''
        items.append(
            f'<item id="{item_id}" href="{href}" media-type="{media_type}"{props_attr}/>'
        )
    refs = [f'<itemref idref="{idref}"/>' for idref in spine_items]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:identifier id="bookid">test-book-id</dc:identifier>
  </metadata>
  <manifest>
    {''.join(items)}
  </manifest>
  <spine>
    {''.join(refs)}
  </spine>
</package>
"""


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def test_href_resolution_percent_and_traversal():
    """3 chapters: plain href, percent-encoded href, '../' traversal href."""
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "text/ch1.xhtml", "application/xhtml+xml", None),
            ("ch2", "ch%202.xhtml", "application/xhtml+xml", None),
            ("ch3", "../text/ch3.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["ch1", "ch2", "ch3"]
        opf_xml = opf(manifest_items, spine)
        files = {
            "OEBPS/text/ch1.xhtml": "<html><body><p>Chapter one content.</p></body></html>",
            # OPF references "ch%202.xhtml" (percent-encoded space) -> actual
            # member on disk is "OEBPS/ch 2.xhtml" (real space).
            "OEBPS/ch 2.xhtml": "<html><body><p>Chapter two content.</p></body></html>",
            # OPF references "../text/ch3.xhtml" relative to OEBPS/, i.e.
            # normalizes to "text/ch3.xhtml" (no OEBPS/ prefix).
            "text/ch3.xhtml": "<html><body><p>Chapter three content.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        text = extract_epub_text(epub_path)
        check(
            "href_resolution: all 3 chapters present",
            "Chapter one content." in text
            and "Chapter two content." in text
            and "Chapter three content." in text,
            detail=repr(text),
        )


def test_case_variant_href():
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "Ch1.XHTML", "application/xhtml+xml", None),
        ]
        spine = ["ch1"]
        opf_xml = opf(manifest_items, spine)
        files = {
            "OEBPS/ch1.xhtml": "<html><body><p>Case variant content.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        text = extract_epub_text(epub_path)
        check(
            "case_variant_href: chapter extracted",
            "Case variant content." in text,
            detail=repr(text),
        )


def test_paragraph_structure_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "ch1.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["ch1"]
        opf_xml = opf(manifest_items, spine)
        paragraphs = [
            "This is the first paragraph.",
            "This is the second paragraph, a bit longer than the first.",
            "Third paragraph here.",
            "And a fourth and final paragraph.",
        ]
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        files = {
            "OEBPS/ch1.xhtml": f"<html><body>{body}</body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        text = extract_epub_text(epub_path)
        pieces = re.split(r'\n\s*\n', text)
        check(
            "paragraph_structure: N paragraphs -> N pieces via \\n\\s*\\n split",
            len(pieces) == len(paragraphs),
            detail=f"expected {len(paragraphs)} pieces, got {len(pieces)}: {pieces!r}",
        )
        check(
            "paragraph_structure: paragraph text matches in order",
            pieces == paragraphs,
            detail=repr(pieces),
        )


def test_head_title_not_in_body_text():
    """<head><title> is document metadata, not book text.

    HTMLParser reports it through handle_data like any other text, so an
    unskipped <title> gets spliced into the narration. The body's own
    heading must survive untouched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "ch1.xhtml", "application/xhtml+xml", None),
        ]
        opf_xml = opf(manifest_items, ["ch1"])
        files = {
            "OEBPS/ch1.xhtml": (
                "<html><head><title>Document Metadata Title</title>"
                "<meta charset=\"utf-8\"/></head><body>"
                "<h1>3: Initial Reconnaissance</h1>"
                "<p>The body paragraph survives.</p>"
                "</body></html>"
            ),
        }
        make_epub(epub_path, opf_xml, files)
        text = extract_epub_text(epub_path)
        check(
            "head_title: <title> text is not extracted",
            "Document Metadata Title" not in text,
            detail=repr(text),
        )
        check(
            "head_title: body heading survives",
            "3: Initial Reconnaissance" in text,
            detail=repr(text),
        )
        check(
            "head_title: body paragraph survives",
            "The body paragraph survives." in text,
            detail=repr(text),
        )


def test_nav_doc_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("nav", "nav.xhtml", "application/xhtml+xml", "nav"),
            ("ch1", "ch1.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["nav", "ch1"]
        opf_xml = opf(manifest_items, spine)
        files = {
            "OEBPS/nav.xhtml": (
                "<html><body><nav epub:type='toc'>"
                "<h1>Table of Contents</h1><ol><li>Chapter 1</li></ol>"
                "</nav></body></html>"
            ),
            "OEBPS/ch1.xhtml": "<html><body><p>Real chapter content.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        text = extract_epub_text(epub_path)
        check(
            "nav_doc_skipped: nav content absent, chapter content present",
            "Table of Contents" not in text and "Real chapter content." in text,
            detail=repr(text),
        )


def test_broken_href_raises_valueerror():
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "does-not-exist.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["ch1"]
        opf_xml = opf(manifest_items, spine)
        files = {
            # Deliberately no file matching "does-not-exist.xhtml" under any
            # fallback (normalized, case-insensitive, or basename).
            "OEBPS/unrelated.xhtml": "<html><body><p>Unrelated.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        raised = False
        try:
            extract_epub_text(epub_path)
        except ValueError:
            raised = True
        except Exception as e:
            check(
                "broken_href: raises ValueError (not some other exception)",
                False,
                detail=f"raised {type(e).__name__}: {e}",
            )
            return
        check("broken_href: raises ValueError instead of silently skipping", raised)


def test_non_html_spine_item_skipped():
    """A spine may legally reference non-text content (e.g. an SVG cover
    page). That manifest item DOES exist -- it must be skipped with a
    notice, not raise, and must not derail extraction of the real chapter.
    """
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("cover", "cover.svg", "image/svg+xml", None),
            ("ch1", "ch1.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["cover", "ch1"]
        opf_xml = opf(manifest_items, spine)
        files = {
            "OEBPS/cover.svg": (
                '<svg xmlns="http://www.w3.org/2000/svg">'
                '<title>Cover</title></svg>'
            ),
            "OEBPS/ch1.xhtml": "<html><body><p>Real chapter content.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        try:
            text = extract_epub_text(epub_path)
        except Exception as e:
            check(
                "non_html_spine_item: extraction succeeds without raising",
                False,
                detail=f"raised {type(e).__name__}: {e}",
            )
            return
        check("non_html_spine_item: extraction succeeds without raising", True)
        check(
            "non_html_spine_item: chapter text present",
            "Real chapter content." in text,
            detail=repr(text),
        )


def test_missing_idref_raises_valueerror():
    """A spine idref that has NO corresponding manifest item at all (as
    opposed to a manifest item that exists but is non-HTML) must still
    raise -- this is the genuinely-missing-item case, distinct from the
    legally-non-text-item case above.
    """
    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            ("ch1", "ch1.xhtml", "application/xhtml+xml", None),
        ]
        spine = ["ch1", "ghost"]  # "ghost" has no manifest entry whatsoever
        opf_xml = opf(manifest_items, spine)
        files = {
            "OEBPS/ch1.xhtml": "<html><body><p>Chapter content.</p></body></html>",
        }
        make_epub(epub_path, opf_xml, files)
        raised = False
        try:
            extract_epub_text(epub_path)
        except ValueError:
            raised = True
        except Exception as e:
            check(
                "missing_idref: raises ValueError (not some other exception)",
                False,
                detail=f"raised {type(e).__name__}: {e}",
            )
            return
        check("missing_idref: raises ValueError for idref absent from manifest", raised)


def main():
    tests = [
        test_href_resolution_percent_and_traversal,
        test_case_variant_href,
        test_paragraph_structure_preserved,
        test_head_title_not_in_body_text,
        test_nav_doc_skipped,
        test_broken_href_raises_valueerror,
        test_non_html_spine_item_skipped,
        test_missing_idref_raises_valueerror,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            check(t.__name__, False, detail=traceback.format_exc())

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
