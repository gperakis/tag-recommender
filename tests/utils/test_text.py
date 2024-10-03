import pytest

from tag_recommender.utils.text import (
    normalize_hashtags,
    preprocess_corpus,
    to_snake_case,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("ThisIsPascalCase", "this_is_pascal_case"),
        ("camelCaseExample", "camel_case_example"),
        ("already_snake_case", "already_snake_case"),
        ("MixedCase Example", "mixed_case_example"),
        ("some-text-with-hyphens", "some_text_with_hyphens"),
        ("  leading and trailing spaces  ", "leading_and_trailing_spaces"),
        ("PascalCamel case_with_snake", "pascal_camel_case_with_snake"),
        ("Multiple   Spaces--and__underscores", "multiple_spaces_and_underscores"),
        ("", ""),
    ],
)
def test_to_snake_case(input_text, expected_output):
    assert to_snake_case(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("tagOne,tagTwo,tagThree", ["tag_one", "tag_two", "tag_three"]),
        ("PascalCaseTag,anotherTag", ["pascal_case_tag", "another_tag"]),
        ("singleTag", ["single_tag"]),
        ("snake_case_tag1,snake_case_tag2", ["snake_case_tag1", "snake_case_tag2"]),
        ("  tag with spaces ,another_tag   ", ["tag_with_spaces", "another_tag"]),
        (
            "dash-tag, multiple-tags-with-dashes",
            ["dash_tag", "multiple_tags_with_dashes"],
        ),
        ("tag with  spaces and-hyphens", ["tag_with_spaces_and_hyphens"]),
        ("", []),
    ],
)
def test_normalize_hashtags(input_text, expected_output):
    assert normalize_hashtags(input_text) == expected_output


@pytest.mark.parametrize(
    "input_corpus, expected_output",
    [
        (["tagOne,tagTwo,tagThree"], [["tag_one", "tag_two", "tag_three"]]),
        (
            ["PascalCaseTag,anotherTag", "thirdTag,fourth_tag"],
            [["pascal_case_tag", "another_tag"], ["third_tag", "fourth_tag"]],
        ),
        (["singleTag"], [["single_tag"]]),
        (
            ["snake_case_tag1,snake_case_tag2", "moreTags,differentTag"],
            [["snake_case_tag1", "snake_case_tag2"], ["more_tags", "different_tag"]],
        ),
        ([""], [[]]),
        ([" tagWithSpaces , another_tag "], [["tag_with_spaces", "another_tag"]]),
    ],
)
def test_preprocess_corpus(input_corpus, expected_output):
    assert preprocess_corpus(input_corpus) == expected_output
