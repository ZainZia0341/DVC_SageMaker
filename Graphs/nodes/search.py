import boto3
import json
from ..Graph_State import State

def count_matches(analysis_text, tags):
    """
    Count how many of the tags appear in the analysis text.
    For simplicity, we check by converting both to lowercase.
    """
    count = 0
    analysis_text = str(analysis_text).lower()
    for tag in tags:
        if tag.lower() in analysis_text:
            count += 1
    return count

def file_search_node(state: State):
    """
    Searches the DynamoDB PublicFiles table for items whose 'analysis' field contains at least one of the provided tags.
    For each found file, it counts how many tags match.
    Then it sorts the files from highest to lowest match count and applies an optional threshold filter.
    Finally, it returns a list of dictionaries for the files.

    For example, the returned result might look like:
    {
      "files": [
         {"file_name": "FileA.pdf", "matched_tags": 5},
         {"file_name": "FileB.pdf", "matched_tags": 4}
      ]
    }
    """
    print("------------------------ file search node -----------------------------")
    print("State received:", state)
    print("file_search_node => searching for tags:", state["tags"])
    
    # 1) Connect to DynamoDB
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table("PublicFiles")  # Adjust name if needed

    # 2) Get the tags from state. If state["tags"] is already a list, use it.
    tags_list = state["tags"] if isinstance(state["tags"], list) else []
    
    # If no tags, return empty result.
    if not tags_list:
        print("No tags provided, returning empty result.")
        return {"files": []}

    # 3) Build filter expressions, one per tag.
    filter_expressions = []
    expression_values = {}
    for i, t in enumerate(tags_list):
        filter_expressions.append(f"contains(#analysis, :tag{i})")
        expression_values[f":tag{i}"] = t
    
    # Join the expressions with OR; so only one tag needs to match.
    final_filter_expr = " OR ".join(filter_expressions)
    print("Final filter expression:", final_filter_expr)
    
    # 4) Scan the table with our filter.
    response = table.scan(
        ProjectionExpression="file_name, analysis",  # Return only these attributes.
        FilterExpression=final_filter_expr,
        ExpressionAttributeValues=expression_values,
        ExpressionAttributeNames={"#analysis": "analysis"}
    )

    items = response.get("Items", [])
    print("---------- files found from dynamodb --------------")
    print(items)
    
    # 5) For each matching item, count how many tags matched.
    found_files = []
    for item in items:
        if "file_name" in item and "analysis" in item:
            num_matches = count_matches(item["analysis"], tags_list)
            found_files.append({"file_name": item["file_name"], "matched_tags": num_matches})
    
    # 6) (Optional Filter) Get threshold value from state; default to 0 if not provided.
    threshold = state.get("match_threshold", 0)
    if threshold:
        # Only keep files that match at least the threshold.
        found_files = [f for f in found_files if f["matched_tags"] >= threshold]
    
    # 7) Sort the files by matched_tags from highest to lowest.
    sorted_files = sorted(found_files, key=lambda x: x["matched_tags"], reverse=True)
    
    print("Found files with match counts (sorted):", sorted_files)
    return {"files": sorted_files}
