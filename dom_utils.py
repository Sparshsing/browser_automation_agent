import re
from collections import deque
from bs4 import BeautifulSoup, Comment, NavigableString
import crawl4ai.content_scraping_strategy as scrapper
from utils import create_custom_logger

logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)


# --- Configuration ---
INTERACTIVE_TAGS = {
    'a', 'button', 'input', 'select', 'textarea', 'label', 'option',
    'faceplate-search-input' # Custom interactive tag example
}
INTERACTIVE_ROLES = {
    'button', 'link', 'checkbox', 'radio', 'menuitem', 'tab', 'slider',
    'spinbutton', 'switch', 'textbox', 'listbox', 'combobox', 'searchbox'
}
CONTEXT_TAGS = {
    'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    # 'table', 'thead', 'tbody', 'tr', 'th', 'td' # Optional
}
ALLOWED_ATTRIBUTES = {
    'id', 'class', 'name', 'role', 'href', 'src', 'alt', 'placeholder',
    'value', 'type', 'for', 'title', 'disabled', 'checked', 'selected',
    'data-testid', 'data-cy', 'data-test', 'aria-label', 'aria-labelledby',
    'aria-disabled'
    # Add other critical data-* or aria-* if needed by name
}
MAX_ATTR_LENGTH = 150
TEMP_KEEP_ATTR = '_keep_this_tag_marker_'

# --- Tags to remove unconditionally BEFORE processing ---
TAGS_TO_REMOVE_FIRST = {'script', 'style', 'link', 'meta', 'noscript', 'head', 'svg'} # Added svg, head

# --- Helper Function for Attribute Filtering ---
def filter_attributes(tag):
    if not hasattr(tag, 'attrs'):
        return

    original_attrs = dict(tag.attrs)
    filtered_attrs = {}
    for attr, value in original_attrs.items():
        if attr == TEMP_KEEP_ATTR:
            filtered_attrs[attr] = value
            continue

        attr_lower = attr.lower()
        is_allowed = False
        if attr_lower in ALLOWED_ATTRIBUTES or \
           attr_lower.startswith('data-') or \
           attr_lower.startswith('aria-'):
           is_allowed = True

        if is_allowed:
            if isinstance(value, (str, list)) and not isinstance(value, bool): # Check length only for str/list
                 val_str = "".join(value) if isinstance(value, list) else value
                 if len(val_str) > MAX_ATTR_LENGTH:
                    filtered_attrs[attr] = val_str[:MAX_ATTR_LENGTH] + "...[truncated]"
                 else:
                    filtered_attrs[attr] = value # Keep original type (str or list)
            else:
                 filtered_attrs[attr] = value # Keep bools or short values

    tag.attrs = filtered_attrs

import time
# --- Main Reduction Function (Final Version) ---
def get_interactive_dom(html_content: str) -> str:
    """
    Reduces HTML by:
    1. Removing unwanted tags (script, style, meta, etc.) first.
    2. Keeping interactive/contextual elements, their ancestors, and descendants.
    3. Filtering attributes on remaining elements.

    Args:
        html_content: The original HTML string.

    Returns:
        A string containing the reduced HTML structure.
    """
    if not html_content:
        return ""
    t0 = time.time()
    soup = BeautifulSoup(html_content, 'lxml')
    elements_to_process_for_ancestors = []

    # --- Step 0: Initial Cleanup ---
    # Remove comments first
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # Remove unwanted tags entirely before marking
    for tag_name in TAGS_TO_REMOVE_FIRST:
        for tag in soup.find_all(tag_name):
            tag.decompose() # Remove the tag and its contents

    t1 = time.time()
    # --- Step 1: Mark initial seed elements ---
    all_tags_initial = soup.find_all(True) # Find remaining tags
    for tag in all_tags_initial:
        marked = False
        # Check tag name
        if tag.name in INTERACTIVE_TAGS or tag.name in CONTEXT_TAGS:
            tag[TEMP_KEEP_ATTR] = 'seed'
            marked = True

        # Check role
        role = tag.attrs.get('role', '').lower()
        if role in INTERACTIVE_ROLES:
            tag[TEMP_KEEP_ATTR] = 'seed'
            marked = True

        # Check for input-like attributes (heuristic)
        if tag.has_attr('type') and tag.attrs.get('type') in ['text', 'search', 'email', 'password', 'url', 'tel', 'number', 'checkbox', 'radio', 'submit', 'reset', 'button']:
             tag[TEMP_KEEP_ATTR] = 'seed'
             marked = True
        if tag.has_attr('placeholder') and tag.name not in ['div', 'span']:
             tag[TEMP_KEEP_ATTR] = 'seed'
             marked = True

        # Check label 'for'
        if tag.name == 'label' and tag.has_attr('for'):
            tag[TEMP_KEEP_ATTR] = 'seed'
            marked = True

        if marked and tag not in elements_to_process_for_ancestors:
            elements_to_process_for_ancestors.append(tag)

    t2 = time.time()
    # --- Step 2: Mark ancestors of seed elements ---
    queue = deque(elements_to_process_for_ancestors)
    processed_for_ancestors = set(elements_to_process_for_ancestors)

    while queue:
        current = queue.popleft()
        parent = getattr(current, 'parent', None)
        if parent and getattr(parent, 'name', None) and parent.name != '[document]' and parent not in processed_for_ancestors:
            # Only mark parent if it doesn't already have a 'seed' marker
            if TEMP_KEEP_ATTR not in parent.attrs or parent.attrs[TEMP_KEEP_ATTR] != 'seed':
                 parent[TEMP_KEEP_ATTR] = 'ancestor'
            processed_for_ancestors.add(parent)
            queue.append(parent)

    t3 = time.time()
    # --- Step 3: Mark all descendants of *any* marked element ---
    marked_elements_roots = soup.find_all(attrs={TEMP_KEEP_ATTR: True})

    for root in marked_elements_roots:
        # Don't descend into already removed tag types (though they should be gone)
        if root.name in TAGS_TO_REMOVE_FIRST: continue

        for descendant in root.find_all(True):
            # Don't mark descendants that should have been removed initially
            if descendant.name in TAGS_TO_REMOVE_FIRST: continue
            # Mark descendant if not already marked
            if TEMP_KEEP_ATTR not in descendant.attrs:
                 descendant[TEMP_KEEP_ATTR] = 'descendant'

    t4 = time.time()
    # --- Step 4: Remove elements NOT marked ---
    all_tags_final_check = soup.find_all(True)
    for tag in all_tags_final_check:
        if tag.parent is None and tag is not soup and tag.name != 'html':
            continue
        if TEMP_KEEP_ATTR not in tag.attrs:
            # Double check it's not a tag we meant to remove anyway
            if tag.name not in TAGS_TO_REMOVE_FIRST:
                 tag.decompose()
            # If it IS a tag we meant to remove first but somehow survived, remove it now.
            elif tag.name in TAGS_TO_REMOVE_FIRST:
                 tag.decompose()

    t5 = time.time()        
    # --- Step 5: Clean up remaining elements ---
    remaining_tags = soup.find_all(True)
    for tag in remaining_tags:
        # Comment removal already done in Step 0
        # Filter attributes
        filter_attributes(tag)

        # Remove the temporary marker
        if TEMP_KEEP_ATTR in tag.attrs:
            del tag[TEMP_KEEP_ATTR]

    t6 = time.time()
    # --- Step 6: Remove empty/whitespace text nodes (optional) ---
    for element in soup.descendants:
         if isinstance(element, NavigableString) and element.strip() == "":
             if element.parent and element.parent.name not in ['pre', 'textarea']:
                 element.extract()

    t7 = time.time()
    # --- Step 7: Final Output ---
    reduced_html = str(soup.body) if soup.body else str(soup) # Often just want body content
    # Optional: more aggressive whitespace removal
    # reduced_html = re.sub(r'>\s*<', '><', reduced_html)
    # reduced_html = re.sub(r'\s{2,}', ' ', reduced_html).strip()
    t8 = time.time()

    logger.debug(f"Time taken to get interactive DOM: {t8 - t0} seconds")
    # print(f"Time taken to mark initial seed elements: {t2 - t1} seconds")
    # print(f"Time taken to mark ancestors of seed elements: {t3 - t2} seconds")
    # print(f"Time taken to remove elements NOT marked: {t5 - t4} seconds")
    # print(f"Time taken to clean up remaining elements: {t6 - t5} seconds")
    # print(f"Time taken to mark all descendants of *any* marked element: {t4 - t3} seconds")
    # print(f"Time taken to final output: {t8 - t7} seconds")
    return reduced_html



def get_simplified_dom(html: str, url: str) -> str:
    """
    Get a simplified DOM representation of the current page
    """
    scrapping_strategy = scrapper.WebScrapingStrategy()
    scrap_result = scrapping_strategy._scrap(url=url, html=html)
    simplified_dom = scrap_result['cleaned_html']
    return simplified_dom



async def get_shadow_dom(locator):
    """
    Get the shadow DOM of the element.

    Args:
        locator: The playwrightlocator of the element to get the shadow DOM of.
        
    Returns:
        The shadow DOM of the element.
    """
    inner_html = await locator.nth(0).evaluate("""
            element => {
                if (element.shadowRoot && element.shadowRoot.mode === 'open') {
                    // Element has an open shadow root, return its inner HTML
                    return element.shadowRoot.innerHTML;
                } else {
                    // No open shadow root, return the element's standard inner HTML
                    return element.innerHTML;
                }
            }
        """)
    return inner_html


async def get_full_dom_with_shadow(page) -> str:
    """
    Get the full DOM with shadow DOM.
    """
    full_html = await page.evaluate("""
        () => {
            function serializeNode(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const tagName = node.tagName.toLowerCase();
                    const attrs = Array.from(node.attributes).map(attr => 
                        `${attr.name}="${attr.value}"`
                    ).join(" ");
                    let openingTag = `<${tagName}${attrs ? ' ' + attrs : ''}>`;
                    let closingTag = `</${tagName}>`;

                    // Shadow root detection
                    let shadowHtml = "";
                    if (node.shadowRoot && node.shadowRoot.mode === "open") {
                        const shadowContent = Array.from(node.shadowRoot.childNodes)
                            .map(serializeNode).join("");
                        shadowHtml = `<template shadowroot="open">${shadowContent}</template>`;
                    }

                    // Normal child nodes
                    const childrenHtml = Array.from(node.childNodes).map(serializeNode).join("");

                    return `${openingTag}${shadowHtml}${childrenHtml}${closingTag}`;
                } else if (node.nodeType === Node.TEXT_NODE) {
                    return node.textContent;
                } else if (node.nodeType === Node.COMMENT_NODE) {
                    return `<!--${node.textContent}-->`;
                } else {
                    return "";
                }
            }

            return "<!DOCTYPE html>" + serializeNode(document.documentElement);
        }
    """)
    return full_html

