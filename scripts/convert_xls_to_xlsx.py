"""
将 .xls 文件转换为 .xlsx 格式
使用 win32com (仅在 Windows 上可用)
"""
import os
import sys

def convert_xls_to_xlsx(xls_path: str, xlsx_path: str = None):
    """
    将 .xls 文件转换为 .xlsx 格式

    Args:
        xls_path: 输入的 .xls 文件路径
        xlsx_path: 输出的 .xlsx 文件路径（可选）
    """
    try:
        import win32com.client

        if xlsx_path is None:
            xlsx_path = xls_path.replace('.xls', '.xlsx')

        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        workbook = excel.Workbooks.Open(os.path.abspath(xls_path))
        workbook.SaveAs(os.path.abspath(xlsx_path), FileFormat=51)  # 51 = xlsx
        workbook.Close()
        excel.Quit()

        print(f"✅ 成功转换: {xls_path} -> {xlsx_path}")
        return True

    except ImportError:
        print("❌ win32com 未安装，正在安装...")
        os.system("pip install pywin32")
        print("请重新运行脚本")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


if __name__ == '__main__':
    xls_file = 'skills/topology/assets/OS.xls'
    xlsx_file = 'skills/topology/assets/OS.xlsx'

    print("="*60)
    print("XLS to XLSX Converter")
    print("="*60)

    if convert_xls_to_xlsx(xls_file, xlsx_file):
        print(f"\n✅ 转换完成！")
        print(f"原始文件: {xls_file}")
        print(f"转换后: {xlsx_file}")
