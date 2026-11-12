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
