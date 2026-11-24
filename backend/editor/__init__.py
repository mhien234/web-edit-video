"""Trình dựng (editor) theo cách Vidlab.

Mọi thay đổi trên timeline, của người hay của AI agent, đều là một *lệnh* đi qua
EditorCore (core.py). Mỗi lệnh tạo một phiên bản mới trong SQLite, nên:
- hoàn tác / làm lại từng bước,
- người và agent cùng sửa một project mà không ghi đè nhau.
"""
