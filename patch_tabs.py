import os

tabs = [
    ('schema_discovery', 'schema_discovery'),
    ('entity_resolution', 'entity_resolution'),
    ('entity_graph', 'entity_graph'),
    ('data_quality', 'data_quality'),
    ('fraud_detection', 'fraud'),
    ('aml_detection', 'aml'),
    ('risk_clusters', 'fraud_rings'),
    ('investigations', 'generic'),
    ('case_management', 'case_management'),
    ('alerts', 'generic'),
    ('reports', 'generic')
]

for tab_file, domain in tabs:
    path = f'tabs/{tab_file}.py'
    if not os.path.exists(path): continue
    
    with open(path, 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'dummy_btn.click(fn=lambda: generate_dummy_data(' in line:
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            new_lines.append(spaces + 'dummy_count = gr.Dropdown(choices=["15", "100", "500", "1000", "5000", "10000"], value="15", label="Records to Generate")')
            new_lines.append(spaces + f'dummy_btn.click(fn=lambda n: generate_dummy_data("{domain}", int(n)), inputs=dummy_count, outputs=ds_upload)')
        else:
            new_lines.append(line)
            
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f'Patched {path}')
