import unittest

from source_normalization import strip_publisher_matter


class StripPublisherMatterTest(unittest.TestCase):
    """Official epubs carry a publisher colophon at one or both ends. It was
    being narrated: eminence04 opened with 16 entries of copyright page, and
    otherside07 closed with 12 including a street address and a URL."""

    FRONT = (
        "Copyright\n\n"
        "The Eminence in Shadow 04\n\n"
        "DAISUKE AIZAWA\n\n"
        "Translation by Nathaniel Hiroshi Thrasher\n\n"
        "This book is a work of fiction. Names, characters, places, and "
        "incidents are the product of the author's imagination.\n\n"
        "©Daisuke Aizawa 2021\n\n"
        "First published in Japan in 2021 by KADOKAWA CORPORATION, Tokyo.\n\n"
        "English translation © 2022 by Yen Press, LLC\n\n"
        "Yen On\n\n150 West 30th Street, 19th Floor\n\nNew York, NY 10001\n\n"
        "ISBN 978-1-9753-3776-2\n\n"
    )
    STORY = ("The snow fell steadily over the quiet road.\n\n"
             "Shadow considered the situation before speaking again.\n\n"
             "“I am the eminence in shadow,” he said.\n\n") * 12
    BACK = (
        "\nCopyright © 2021 Iori Miyazawa\n\n"
        "All rights reserved.\n\n"
        "First published in Japan in 2021 by Hayakawa Publishing Corporation\n\n"
        "English translation © 2021 J-Novel Club LLC\n\n"
        "J-Novel Club LLC\n\nj-novel.club\n\n"
        "Ebook edition 1.0: May 2022\n"
    )

    def test_front_matter_is_removed(self):
        text, report = strip_publisher_matter(self.FRONT + self.STORY)
        self.assertNotIn("KADOKAWA", text)
        self.assertNotIn("150 West 30th Street", text)
        self.assertTrue(text.lstrip().startswith("The snow fell"))
        self.assertGreater(report["front_paragraphs"], 0)

    def test_back_matter_is_removed(self):
        text, report = strip_publisher_matter(self.STORY + self.BACK)
        self.assertNotIn("j-novel.club", text)
        self.assertNotIn("Ebook edition", text)
        self.assertGreater(report["back_paragraphs"], 0)

    def test_both_ends_at_once(self):
        text, report = strip_publisher_matter(self.FRONT + self.STORY + self.BACK)
        self.assertNotIn("©", text)
        self.assertIn("eminence in shadow", text)
        self.assertGreater(report["front_paragraphs"], 0)
        self.assertGreater(report["back_paragraphs"], 0)

    def test_story_is_untouched_when_there_is_no_colophon(self):
        text, report = strip_publisher_matter(self.STORY)
        self.assertEqual(text, self.STORY)
        self.assertEqual(report["front_paragraphs"], 0)
        self.assertEqual(report["back_paragraphs"], 0)

    def test_story_mentioning_copyright_mid_book_is_kept(self):
        # A single legal-sounding line inside the story must not trigger a strip
        # from the middle: only runs anchored at an end are removed.
        text, _ = strip_publisher_matter(
            self.STORY + "\n\nHe read the copyright page aloud, bored.\n\n" + self.STORY)
        self.assertIn("read the copyright page aloud", text)

    def test_a_book_that_is_all_colophon_is_left_alone(self):
        # Refuse to strip everything: that is a broken source, not front matter.
        text, report = strip_publisher_matter(self.FRONT)
        self.assertEqual(text, self.FRONT)
        self.assertEqual(report["front_paragraphs"], 0)


if __name__ == "__main__":
    unittest.main()
