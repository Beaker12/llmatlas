"""Offline tests of the comparison logic in tools/verify_arxiv.py, with canned API responses."""
import json

from tools import verify_arxiv as va

ATOM = """<feed>
<entry>
  <id>http://arxiv.org/abs/2003.08271v3</id>
  <title>Pre-trained Models for Natural Language
  Processing: A Survey</title>
  <author><name>Xipeng Qiu</name></author>
  <author><name>Tianxiang Sun</name></author>
  <arxiv:journal_ref xmlns:arxiv="x">SCIENCE CHINA Technological Sciences 63,
  1872-1897 (2020)</arxiv:journal_ref>
  <arxiv:doi xmlns:arxiv="x">10.1007/s11431-020-1647-3</arxiv:doi>
</entry>
<entry>
  <id>http://arxiv.org/abs/1712.00409v1</id>
  <title>Deep Learning Scaling is Predictable, Empirically</title>
  <author><name>Joel Hestness</name></author>
  <author><name>Md. Mostafa Ali Patwary</name></author>
  <author><name>Hal Daumé III</name></author>
</entry>
</feed>"""

CROSSREF = json.dumps({"message": {
    "title": ["Attention is all you need"],
    "container-title": ["Advances in Neural Information Processing Systems"],
    "volume": "30", "page": "5998-6008",
    "author": [{"family": "Vaswani"}, {"family": "Shazeer"}],
}})


def parse(bib):
    return dict(va.parse_entries(bib))


def test_atom_parsing():
    records = va.parse_atom(ATOM)
    assert set(records) == {"2003.08271", "1712.00409"}
    qiu = records["2003.08271"]
    assert qiu["title"] == "Pre-trained Models for Natural Language Processing: A Survey"
    assert qiu["authors"] == ["Xipeng Qiu", "Tianxiang Sun"]
    assert qiu["journal_ref"].startswith("SCIENCE CHINA Technological Sciences")
    assert qiu["doi"] == "10.1007/s11431-020-1647-3"
    assert records["1712.00409"]["journal_ref"] is None


def test_journal_mismatch_against_the_journal_reference():
    fields = parse("""@article{qiu,
  title={Pre-trained models for natural language processing: A survey},
  author={Qiu, Xipeng and Sun, Tianxiang},
  journal={Science China Information Sciences},
  volume={63},
  pages={1--25},
  eprint={2003.08271}
}""")["qiu"]
    problems = va.compare_arxiv(fields, va.parse_atom(ATOM)["2003.08271"])
    assert any(p.startswith("journal:") for p in problems)
    assert any(p.startswith("pages:") for p in problems)
    assert not any(p.startswith("volume:") for p in problems)


def test_matching_entry_has_no_problems():
    fields = parse("""@article{qiu,
  title={Pre-trained models for NLP: a survey, with a different published title},
  author={Qiu, Xipeng and Sun, Tianxiang},
  journal={SCIENCE CHINA Technological Sciences},
  volume={63},
  pages={1872--1897},
  eprint={2003.08271}
}""")["qiu"]
    assert va.compare_arxiv(fields, va.parse_atom(ATOM)["2003.08271"]) == []


def test_author_list_mismatch_is_reported():
    fields = parse("""@article{h,
  title={Deep learning scaling is predictable, empirically},
  author={Hestness, Joel and Patwary, Md and Ali, Mostafa and Daum{\\'e} III, Hal},
  journal={arXiv preprint arXiv:1712.00409},
  eprint={1712.00409}
}""")["h"]
    problems = va.compare_arxiv(fields, va.parse_atom(ATOM)["1712.00409"])
    assert problems == ["author count 4 in the entry, 3 in the record"]


def test_arxiv_preprint_venue_is_not_compared_and_accents_match():
    fields = parse("""@article{h,
  title={Deep learning scaling is predictable, empirically},
  author={Hestness, Joel and Patwary, Md. Mostafa Ali and Daum{\\'e} III, Hal},
  journal={arXiv preprint arXiv:1712.00409},
  eprint={1712.00409}
}""")["h"]
    assert va.compare_arxiv(fields, va.parse_atom(ATOM)["1712.00409"]) == []


def test_title_is_compared_only_without_a_venue():
    preprint = parse("""@misc{x,
  title={A completely different title about something else},
  author={Hestness, Joel and Patwary, Mostafa and Daum{\\'e} III, Hal},
  eprint={1712.00409}
}""")["x"]
    problems = va.compare_arxiv(preprint, va.parse_atom(ATOM)["1712.00409"])
    assert [p.split(":")[0] for p in problems] == ["title"]


def test_others_truncates_the_author_comparison():
    fields = {"author": "Hestness, Joel and others"}
    assert va.authors_match(fields["author"], ["Joel Hestness", "A B", "C D"]) == []


def test_punctuation_only_record_authors_are_ignored():
    assert va.authors_match("{Qwen Team} and Yang, An", ["Qwen", ":", "An Yang"]) == []


def test_missing_record_is_reported():
    assert va.compare_arxiv({"title": "x"}, None) == ["arXiv has no record for this id"]


def test_crossref_comparison():
    record = va.parse_crossref(CROSSREF)
    good = {"author": "Vaswani, Ashish and Shazeer, Noam",
            "booktitle": "Advances in Neural Information Processing Systems 30",
            "volume": "30", "pages": "5998--6008"}
    assert va.compare_crossref(good, record) == []
    bad = dict(good, booktitle="Proceedings of the Conference on Robot Learning",
               pages="1--9", author="Vaswani, Ashish and Parmar, Niki")
    problems = va.compare_crossref(bad, record)
    assert [p.split(":")[0].split(" ")[0] for p in problems] == ["author", "journal", "pages"]


def test_journal_reference_parsing():
    assert va.split_journal_ref("SCIENCE CHINA Technological Sciences 63, 1872-1897 (2020)") == (
        "SCIENCE CHINA Technological Sciences", "63", "1872-1897")
    assert va.split_journal_ref("Transactions of the ACL 2022") == (
        "Transactions of the ACL", None, None)


def test_undecidable_venues_count_as_a_match():
    assert va.same_venue("Proceedings of the 37th International Conference on Machine Learning",
                         "Published on ICML")
    assert va.same_venue("Proceedings of the IEEE conference on computer vision and pattern"
                         " recognition",
                         "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)")
    assert not va.same_venue("Science China Information Sciences",
                             "SCIENCE CHINA Technological Sciences")


# --- title rule (item 2) ---------------------------------------------------------------------


def test_preprint_string_in_journal_still_gets_the_title_check():
    fields = parse("""@article{x,
  title={A completely different title about something else},
  author={Hestness, Joel and Patwary, Mostafa and Daum{\\'e} III, Hal},
  journal={arXiv preprint arXiv:1712.00409},
  eprint={1712.00409}
}""")["x"]
    problems = va.compare_arxiv(fields, va.parse_atom(ATOM)["1712.00409"])
    assert [p.split(":")[0] for p in problems] == ["title"]


def test_published_entry_is_not_checked_on_title():
    fields = parse("""@article{x,
  title={A different published title},
  author={Hestness, Joel and Patwary, Mostafa and Daum{\\'e} III, Hal},
  journal={Journal of Things},
  eprint={1712.00409}
}""")["x"]
    assert va.compare_arxiv(fields, va.parse_atom(ATOM)["1712.00409"]) == []


# --- venues (item 3) -------------------------------------------------------------------------


def test_venue_abbreviations_and_acronyms():
    assert va.same_venue("Nature Machine Intelligence", "Nat. Mach. Intell.")
    assert va.same_venue("J. Mach. Learn. Res.", "Journal of Machine Learning Research")
    assert va.same_venue("JMLR", "Journal of Machine Learning Research")
    assert va.same_venue("TACL", "Transactions of the Association for Computational Linguistics")
    assert not va.same_venue("Neural Computation", "Neural Networks")
    assert not va.same_venue("Science China Information Sciences",
                             "Science China Technological Sciences")
    assert not va.same_venue("Nature", "Science")


# --- Crossref through the arXiv DOI (item 4) -------------------------------------------------

NO_REF_ATOM = ATOM.replace(
    '<arxiv:journal_ref xmlns:arxiv="x">SCIENCE CHINA Technological Sciences 63,\n'
    '  1872-1897 (2020)</arxiv:journal_ref>', "")


def test_doi_from_arxiv_is_checked_against_crossref():
    record = va.parse_atom(NO_REF_ATOM)["2003.08271"]
    assert record["journal_ref"] is None and record["doi"] == "10.1007/s11431-020-1647-3"
    fields = parse("""@article{qiu,
  title={Pre-trained models},
  author={Qiu, Xipeng and Sun, Tianxiang},
  journal={Science China Information Sciences},
  eprint={2003.08271}
}""")["qiu"]
    seen = []

    def crossref(doi):
        seen.append(doi)
        return {"title": "", "authors": ["Qiu", "Sun"], "volume": "63", "pages": "1872",
                "journal": "Science China Technological Sciences"}

    problems = va.check_entry(fields, record, crossref, delay=0)
    assert seen == ["10.1007/s11431-020-1647-3"]
    assert any(p.startswith("journal:") for p in problems)


# --- nothing checked, unreachable services (item 5) ------------------------------------------


def test_nothing_checked_is_an_error(tmp_path, capsys):
    bib = tmp_path / "a.bib"
    bib.write_text("@article{k,\n  title={T},\n  author={A, B}\n}\n")
    assert va.main([str(bib)]) == 2
    assert "nothing checked" in capsys.readouterr().out
    bib.write_text("@article{k,\n  title={T},\n  eprint={1712.00409}\n}\n")
    assert va.main([str(bib), "--keys", "nosuchkey"]) == 2
    assert "nothing checked" in capsys.readouterr().out


def test_crossref_outage_is_not_reported_as_a_missing_record():
    fields = {"doi": "10.1/x", "author": "Qiu, Xipeng"}

    def down(doi):
        raise va.Unreachable(doi)

    assert va.check_entry(fields, None, down, delay=0) == ["could not reach Crossref"]
    assert va.check_entry(fields, None, lambda doi: None, delay=0) == [
        "Crossref has no record for this DOI"]


def test_http_404_is_not_found_and_other_failures_are_unreachable(monkeypatch):
    import urllib.error

    def raise_http(code):
        def opener(request, timeout=0):
            raise urllib.error.HTTPError("u", code, "m", {}, None)
        return opener

    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    monkeypatch.setattr(va.urllib.request, "urlopen", raise_http(404))
    assert va.fetch_crossref("10.1/x") is None
    monkeypatch.setattr(va.urllib.request, "urlopen", raise_http(503))
    try:
        va.fetch_crossref("10.1/x")
    except va.Unreachable:
        pass
    else:
        raise AssertionError("expected Unreachable")


# --- whole-surname comparison (item 11) --------------------------------------------------------


def test_surnames_are_compared_whole():
    assert va.authors_match("Li, Wei", ["Wei Ali"])
    assert va.authors_match("Ali, Wei", ["Wei Li"])
    assert va.authors_match("van der Maaten, Laurens", ["Laurens van der Maaten"]) == []
    assert va.authors_match("Daum{\\'e} III, Hal", ["Hal Daumé III"]) == []
    assert va.authors_match("Garc\\'ia M\\'arquez, Gabriel",
                            ["Gabriel García Márquez"]) == []
    assert va.authors_match("Smith-Jones, Ann", ["Ann Smith-Jones"]) == []
    assert va.authors_match("Du, Nan", ["Nan Du"]) == []  # a surname that is also a particle


def test_non_latin_names_are_reported_as_not_comparable():
    problems = va.authors_match("{张}, 三", ["三 张"])
    assert problems and problems[0].startswith("could not compare")


# --- exact venue matching (round 2) ------------------------------------------------------------


def both_ways(a, b):
    assert va.same_venue(a, b) == va.same_venue(b, a), (a, b)
    return va.same_venue(a, b)


def test_one_word_venues_do_not_match_longer_ones():
    for short, long in [("Nature", "Nature Communications"),
                        ("Nature", "Nature Machine Intelligence"),
                        ("Science", "Science Advances"),
                        ("Cell", "Cell Reports")]:
        assert not both_ways(short, long)
    assert both_ways("Nature", "Nature")
    assert not both_ways("Nature", "Science")


def test_abbreviations_still_match_and_near_names_differ():
    for a, b in [("Nat. Mach. Intell.", "Nature Machine Intelligence"),
                 ("J. Mach. Learn. Res.", "Journal of Machine Learning Research"),
                 ("Trans. Assoc. Comput. Linguist.",
                  "Transactions of the Association for Computational Linguistics"),
                 ("IEEE Trans. Pattern Anal. Mach. Intell.",
                  "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
                 ("Neural Comput.", "Neural Computation"),
                 # an abbreviation is inherently ambiguous: this also matches Neural Computing
                 ("Neural Comput.", "Neural Computing")]:
        assert both_ways(a, b), (a, b)
    for a, b in [("Science China Information Sciences", "Science China Technological Sciences"),
                 ("Neural Computation", "Neural Networks"),
                 ("Neural Computation", "Neural Computing and Applications")]:
        assert not both_ways(a, b), (a, b)


def test_empty_or_boilerplate_venue_matches_nothing_else():
    assert not both_ways("Proceedings of the Annual Conference", "Nature")
    assert not both_ways("", "Nature")


def test_initialisms_and_proceedings():
    long_acl = ("Proceedings of the 58th Annual Meeting of the Association for "
                "Computational Linguistics")
    for a, b in [("Proc. ACL", long_acl),
                 ("ACL", "Association for Computational Linguistics"),
                 ("JMLR", "Journal of Machine Learning Research"),
                 ("TACL", "Transactions of the Association for Computational Linguistics"),
                 ("NeurIPS", "Neural Information Processing Systems"),
                 ("ICML", "International Conference on Machine Learning"),
                 ("ICML", "Proceedings of the 37th International Conference on Machine Learning"),
                 ("EMNLP", "Empirical Methods in Natural Language Processing"),
                 ("Symposium on Things 2020", "Things")]:
        assert both_ways(a, b), (a, b)
    for a, b in [("ACL", "Association for Computing Machinery"),
                 ("ICLR", "International Conference on Machine Learning")]:
        assert not both_ways(a, b), (a, b)


# --- arXiv errors (round 2) ------------------------------------------------------------------


def run_main_with_response(monkeypatch, tmp_path, capsys, response):
    bib = tmp_path / "a.bib"
    bib.write_text("@article{k,\n  title={T},\n  eprint={1712.00409}\n}\n")

    def fake_get(url):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(va, "http_get", fake_get)
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    status = va.main([str(bib)])
    return status, capsys.readouterr().out


def test_unreadable_arxiv_response_is_not_a_missing_record(monkeypatch, tmp_path, capsys):
    for response in ["", "<html>Service unavailable</html>", "<feed></feed>"]:
        status, out = run_main_with_response(monkeypatch, tmp_path, capsys, response)
        assert status == 2
        assert "could not read the arXiv response" in out
        assert "no record" not in out


def test_arxiv_404_is_reported_without_a_traceback(monkeypatch, tmp_path, capsys):
    status, out = run_main_with_response(monkeypatch, tmp_path, capsys, va.NotFound("u"))
    assert status == 2
    assert "arXiv endpoint returned 404" in out


def test_unreachable_arxiv_has_the_same_status(monkeypatch, tmp_path, capsys):
    status, out = run_main_with_response(monkeypatch, tmp_path, capsys, va.Unreachable("u"))
    assert status == 2
    assert "could not reach arXiv" in out


def test_journal_reference_cut_down_to_boilerplate_is_not_compared():
    title = ("Proceedings of the 57th Annual Meeting of the Association for "
             "Computational Linguistics")
    journal, volume, pages = va.split_journal_ref(title)
    assert va.compare_publication({"booktitle": title}, journal, volume, pages, "arXiv") == []
