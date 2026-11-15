def paginate(items, page, per_page):
    if per_page < 1:
        raise ValueError("per_page")
    if page < 1:
        raise ValueError("page")
    total_pages = max(1, -(-len(items) // per_page))
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "items": items[start:end],
        "page": page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }
